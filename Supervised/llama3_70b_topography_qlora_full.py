from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from datasets import DatasetDict, load_dataset
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
)
from peft.utils.other import fsdp_auto_wrap_policy
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
)
from trl import SFTTrainer


# ============================================================
# ENVIRONMENT
# ============================================================

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Helps reduce memory fragmentation.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = Path(
    "/workspace/Reviewer_comment/models/"
    "Meta-Llama-3-70B-Instruct"
)

TRAIN_FILE = Path(
    "/workspace/Reviewer_comment/"
    "supervised_data/splits/"
    "topography_train.jsonl"
)

VALIDATION_FILE = Path(
    "/workspace/Reviewer_comment/"
    "supervised_data/splits/"
    "topography_validation.jsonl"
)

OUTPUT_DIR = Path(
    "/workspace/Reviewer_comment/checkpoints/"
    "llama3_70b_topography_qlora_full"
)


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

MAX_SEQUENCE_LENGTH = 512

# Global effective batch:
# 1 sample/GPU × 4 GPUs × 8 accumulation steps = 32
PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
EXPECTED_DISTRIBUTED_PROCESSES = 4

NUM_TRAIN_EPOCHS = 15
LEARNING_RATE = 1.5e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05

LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.10

RANDOM_SEED = 42


# ============================================================
# INITIALIZE ACCELERATE
# ============================================================

accelerator = Accelerator()

local_rank = accelerator.local_process_index
global_rank = accelerator.process_index
world_size = accelerator.num_processes

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available inside the current environment."
    )

torch.cuda.set_device(local_rank)

device_index = torch.cuda.current_device()

print(
    f"[Global rank {global_rank}] "
    f"local_rank={local_rank} | "
    f"device={device_index} | "
    f"GPU={torch.cuda.get_device_name(device_index)}",
    flush=True,
)


# ============================================================
# VALIDATE DISTRIBUTED CONFIGURATION
# ============================================================

if world_size != EXPECTED_DISTRIBUTED_PROCESSES:
    raise RuntimeError(
        "This script requires exactly four distributed processes. "
        f"Accelerate created {world_size} processes."
    )

fsdp_plugin = accelerator.state.fsdp_plugin

if fsdp_plugin is None:
    raise RuntimeError(
        "FSDP was not initialized. Launch this script using an "
        "Accelerate configuration file with FSDP enabled."
    )

global_effective_batch_size = (
    PER_DEVICE_TRAIN_BATCH_SIZE
    * world_size
    * GRADIENT_ACCUMULATION_STEPS
)

if global_effective_batch_size != 32:
    raise RuntimeError(
        "The global effective batch size must be 32, but the "
        f"current configuration produces {global_effective_batch_size}."
    )

if accelerator.is_main_process:
    print("\n" + "=" * 90)
    print("DISTRIBUTED TRAINING CONFIGURATION")
    print("=" * 90)
    print("Distributed method: FSDP")
    print("Distributed processes:", world_size)
    print("Visible GPUs:", torch.cuda.device_count())
    print(
        "Per-device training batch size:",
        PER_DEVICE_TRAIN_BATCH_SIZE,
    )
    print(
        "Gradient accumulation steps:",
        GRADIENT_ACCUMULATION_STEPS,
    )
    print(
        "Global effective batch size:",
        global_effective_batch_size,
    )
    print("Maximum epochs:", NUM_TRAIN_EPOCHS)
    print("Maximum sequence length:", MAX_SEQUENCE_LENGTH)
    print("Learning rate:", LEARNING_RATE)
    print("Weight decay:", WEIGHT_DECAY)
    print("Warmup ratio:", WARMUP_RATIO)
    print("=" * 90 + "\n")


# ============================================================
# VALIDATE PATHS
# ============================================================

required_paths = [
    MODEL_PATH,
    TRAIN_FILE,
    VALIDATION_FILE,
]

for required_path in required_paths:
    if not required_path.exists():
        raise FileNotFoundError(
            "Required path does not exist:\n"
            f"{required_path}"
        )

if accelerator.is_main_process:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

accelerator.wait_for_everyone()


# ============================================================
# REPRODUCIBILITY AND CUDA SETTINGS
# ============================================================

set_seed(RANDOM_SEED)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ============================================================
# LOAD DATASET
# ============================================================

with accelerator.main_process_first():
    dataset: DatasetDict = load_dataset(
        "json",
        data_files={
            "train": str(TRAIN_FILE),
            "validation": str(VALIDATION_FILE),
        },
    )

required_columns = {
    "instruction",
    "input",
    "output",
}

for split_name in ["train", "validation"]:
    available_columns = set(
        dataset[split_name].column_names
    )

    missing_columns = (
        required_columns - available_columns
    )

    if missing_columns:
        raise ValueError(
            f"The {split_name} split is missing columns: "
            f"{sorted(missing_columns)}"
        )

if accelerator.is_main_process:
    print(
        "Training examples:",
        f"{len(dataset['train']):,}",
    )
    print(
        "Validation examples:",
        f"{len(dataset['validation']):,}",
    )


# ============================================================
# TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_PATH),
    use_fast=True,
)

if tokenizer.eos_token is None:
    raise RuntimeError(
        "The tokenizer does not define an EOS token."
    )

# LLaMA models normally do not provide a dedicated padding
# token. The EOS token is therefore used for padding.
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"


# ============================================================
# FORMAT SUPERVISED DATA
# ============================================================

def create_text_column(
    example: dict[str, Any],
) -> dict[str, str]:
    instruction = str(
        example["instruction"]
    ).strip()

    pathology_report = str(
        example["input"]
    ).strip()

    target_code = str(
        example["output"]
    ).strip()

    formatted_text = (
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Pathology report:\n"
        f"{pathology_report}\n\n"
        "### Response:\n"
        f"TOPOGRAPHY_CODE: {target_code}"
        f"{tokenizer.eos_token}"
    )

    return {
        "text": formatted_text,
    }


with accelerator.main_process_first():
    dataset = dataset.map(
        create_text_column,
        remove_columns=dataset["train"].column_names,
        desc="Formatting topography data",
    )

if accelerator.is_main_process:
    print("\nExample formatted training record:\n")
    print(dataset["train"][0]["text"])
    print("\n" + "=" * 90 + "\n")


# ============================================================
# FOUR-BIT QLORA QUANTIZATION
# ============================================================

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,

    # NormalFloat 4-bit quantization.
    bnb_4bit_quant_type="nf4",

    # Double quantization of the quantization constants.
    bnb_4bit_use_double_quant=True,

    # BF16 computation during forward and backward passes.
    bnb_4bit_compute_dtype=torch.bfloat16,

    # BF16 storage is required for the FSDP-QLoRA setup.
    bnb_4bit_quant_storage=torch.bfloat16,
)


# ============================================================
# LOAD MODEL
# ============================================================

print(
    f"[Rank {global_rank}] "
    "Loading LLaMA-3-70B with four-bit NF4 quantization...",
    flush=True,
)

model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_PATH),
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=False,
)

model.config.use_cache = False
model.config.pretraining_tp = 1
model.config.pad_token_id = tokenizer.pad_token_id


# ============================================================
# PREPARE MODEL FOR K-BIT TRAINING
# ============================================================

model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)

# Explicitly enable input gradients because gradient
# checkpointing is used with a frozen quantized backbone.
if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()


# ============================================================
# LORA CONFIGURATION
# ============================================================

lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",

    # Attention projections:
    # query, key, value and output.
    #
    # Feed-forward projections:
    # gate, up and down.
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

training_arguments = TrainingArguments(
    output_dir=str(OUTPUT_DIR),

    # Maximum training duration.
    num_train_epochs=NUM_TRAIN_EPOCHS,

    # Global effective batch size:
    # 1 × 4 GPUs × 8 accumulation = 32.
    per_device_train_batch_size=(
        PER_DEVICE_TRAIN_BATCH_SIZE
    ),
    per_device_eval_batch_size=(
        PER_DEVICE_EVAL_BATCH_SIZE
    ),
    gradient_accumulation_steps=(
        GRADIENT_ACCUMULATION_STEPS
    ),

    # AdamW optimization.
    optim="adamw_torch",
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,

    # Five-percent linear warmup followed by cosine decay.
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type="cosine",

    # BF16 and TF32 computation.
    bf16=True,
    fp16=False,
    tf32=True,

    # Memory-saving configuration.
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={
        "use_reentrant": True,
    },

    # Gradient clipping.
    max_grad_norm=1.0,

    # Evaluate and save after every epoch.
    evaluation_strategy="epoch",
    save_strategy="epoch",

    # Retain the checkpoint with the minimum validation loss.
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    # Keep only the two most recent/best checkpoints.
    save_total_limit=2,

    # Logging configuration.
    logging_strategy="steps",
    logging_steps=10,
    logging_first_step=True,

    # Reproducibility.
    seed=RANDOM_SEED,
    data_seed=RANDOM_SEED,

    # Avoid unnecessary external reporting.
    report_to="none",

    # Dataset and dataloader settings.
    remove_unused_columns=True,
    dataloader_num_workers=0,
    dataloader_pin_memory=True,

    # Do not retain incomplete batches across epochs.
    dataloader_drop_last=False,
)


# ============================================================
# INITIALIZE SFT TRAINER
# ============================================================

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_arguments,

    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],

    peft_config=lora_config,

    dataset_text_field="text",
    max_seq_length=MAX_SEQUENCE_LENGTH,

    # Each pathology report is kept as an independent example.
    packing=False,
)


# ============================================================
# FSDP AUTO-WRAPPING FOR THE LORA MODEL
# ============================================================

if trainer.accelerator.state.fsdp_plugin is not None:
    trainer.accelerator.state.fsdp_plugin.auto_wrap_policy = (
        fsdp_auto_wrap_policy(trainer.model)
    )


# ============================================================
# REPORT TRAINABLE PARAMETERS
# ============================================================

if accelerator.is_main_process:
    trainer.model.print_trainable_parameters()


# ============================================================
# TRAIN
# ============================================================

accelerator.wait_for_everyone()

if accelerator.is_main_process:
    print("\nStarting full FSDP-QLoRA training...\n")

training_result = trainer.train()


# ============================================================
# SAVE BEST MODEL AND TOKENIZER
# ============================================================

accelerator.wait_for_everyone()

# load_best_model_at_end=True restores the checkpoint having
# the lowest validation loss before this save operation.
trainer.save_model(str(OUTPUT_DIR))

if trainer.is_world_process_zero():
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    trainer.save_state()

    train_metrics = training_result.metrics
    trainer.log_metrics(
        "train",
        train_metrics,
    )
    trainer.save_metrics(
        "train",
        train_metrics,
    )

accelerator.wait_for_everyone()


# ============================================================
# FINAL VALIDATION
# ============================================================

validation_metrics = trainer.evaluate(
    eval_dataset=dataset["validation"]
)

if trainer.is_world_process_zero():
    trainer.log_metrics(
        "validation",
        validation_metrics,
    )
    trainer.save_metrics(
        "validation",
        validation_metrics,
    )

    print("\n" + "=" * 90)
    print("TRAINING COMPLETED")
    print("=" * 90)
    print("Best checkpoint:", trainer.state.best_model_checkpoint)
    print("Best validation loss:", trainer.state.best_metric)
    print("Final model directory:", OUTPUT_DIR)
    print("=" * 90)
