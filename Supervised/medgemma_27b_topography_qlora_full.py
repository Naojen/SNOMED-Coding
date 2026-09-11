from __future__ import annotations

from pathlib import Path
import json
import os
from typing import Any

import torch
import transformers
from packaging import version
from accelerate import Accelerator
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
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


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = Path(
    "/workspace/Reviewer_comment/models/"
    "medgemma-27b-text-it"
)


TRAIN_FILE = Path(
    "/workspace/Reviewer_comment/supervised_data/splits/"
    "topography_train.jsonl"
)


VALIDATION_FILE = Path(
    "/workspace/Reviewer_comment/supervised_data/splits/"
    "topography_validation.jsonl"
)


OUTPUT_DIR = Path(
    "/workspace/Reviewer_comment/checkpoints/"
    "medgemma_27b_topography_qlora_full"
)

BEST_ADAPTER_DIR = OUTPUT_DIR / "best_adapter"

TRAINING_SUMMARY_FILE = (
    OUTPUT_DIR
    / "training_summary.json"
)


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

MAX_SEQUENCE_LENGTH = 512
MAXIMUM_EPOCHS = 15

PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 1

# Four GPUs × batch 1 × accumulation 8 = effective batch size 32.
EXPECTED_WORLD_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
EFFECTIVE_BATCH_SIZE = (
    EXPECTED_WORLD_SIZE
    * PER_DEVICE_TRAIN_BATCH_SIZE
    * GRADIENT_ACCUMULATION_STEPS
)

LEARNING_RATE = 1.5e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05

RANDOM_SEED = 42


# ============================================================
# SOFTWARE CHECK
# ============================================================

if version.parse(transformers.__version__) < version.parse("4.50.0"):
    raise RuntimeError(
        "MedGemma/Gemma 3 requires transformers>=4.50.0. "
        f"Current version: {transformers.__version__}"
    )


# ============================================================
# INITIALIZE ACCELERATE AND 4-GPU DDP
# ============================================================

accelerator = Accelerator()

local_rank = accelerator.local_process_index
global_rank = accelerator.process_index
world_size = accelerator.num_processes

torch.cuda.set_device(local_rank)

print(
    f"[Global rank {global_rank}] "
    f"local_rank={local_rank} | "
    f"device={torch.cuda.current_device()} | "
    f"GPU={torch.cuda.get_device_name(local_rank)}",
    flush=True,
)

if world_size != EXPECTED_WORLD_SIZE:
    raise RuntimeError(
        "This training script requires exactly "
        f"{EXPECTED_WORLD_SIZE} distributed processes, "
        f"but Accelerate created {world_size}."
    )

if accelerator.is_main_process:
    print("\n" + "=" * 90)
    print("DISTRIBUTED CONFIGURATION")
    print("=" * 90)

    print("Distributed type:", accelerator.distributed_type)
    print("Distributed processes:", world_size)
    print("Visible GPUs:", torch.cuda.device_count())

    for gpu_index in range(torch.cuda.device_count()):
        free_bytes, total_bytes = torch.cuda.mem_get_info(
            gpu_index
        )

        print(
            f"Local GPU {gpu_index}: "
            f"{torch.cuda.get_device_name(gpu_index)} | "
            f"free={free_bytes / 1024**3:.2f} GiB | "
            f"total={total_bytes / 1024**3:.2f} GiB"
        )

# ============================================================
# CHECK REQUIRED FILES
# ============================================================

for required_path in [
    MODEL_PATH,
    TRAIN_FILE,
    VALIDATION_FILE,
]:
    if not required_path.exists():
        raise FileNotFoundError(
            f"Required path does not exist:\n{required_path}"
        )

if accelerator.is_main_process:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

accelerator.wait_for_everyone()


# ============================================================
# REPRODUCIBILITY
# ============================================================

set_seed(RANDOM_SEED)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ============================================================
# LOAD DATA
# ============================================================

if accelerator.is_main_process:
    print("\n" + "=" * 90)
    print("LOADING TOPOGRAPHY DATASETS")
    print("=" * 90)

with accelerator.main_process_first():
    dataset: DatasetDict = load_dataset(
        "json",
        data_files={
            "train": str(TRAIN_FILE),
            "validation": str(VALIDATION_FILE),
        },
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

    print(
        "Original columns:",
        dataset["train"].column_names,
    )


# ============================================================
# TOKENIZER
# ============================================================

if accelerator.is_main_process:
    print("\n" + "=" * 90)
    print("LOADING MEDGEMMA-27B TOKENIZER")
    print("=" * 90)

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_PATH),
    use_fast=True,
    trust_remote_code=True,
)

if tokenizer.eos_token is None:
    raise RuntimeError(
        "The tokenizer does not define an EOS token."
    )

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

tokenizer.padding_side = "right"

if accelerator.is_main_process:
    print("Vocabulary size:", len(tokenizer))
    print("EOS token:", tokenizer.eos_token)
    print("PAD token:", tokenizer.pad_token)


# ============================================================
# FORMAT DATA
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

    text = (
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Pathology report:\n"
        f"{pathology_report}\n\n"
        "### Response:\n"
        f"TOPOGRAPHY_CODE: {target_code}"
        f"{tokenizer.eos_token}"
    )

    return {
        "text": text,
    }


with accelerator.main_process_first():
    dataset = dataset.map(
        create_text_column,
        remove_columns=dataset["train"].column_names,
        desc="Formatting topography data",
    )

if accelerator.is_main_process:
    print("\nFormatted columns:", dataset["train"].column_names)

    print("\nExample formatted training record:")
    print("-" * 90)
    print(dataset["train"][0]["text"][:1500])
    print("-" * 90)


# ============================================================
# QLORA CONFIGURATION
# ============================================================

if accelerator.is_main_process:
    print("\n" + "=" * 90)
    print("CONFIGURING 4-BIT QLORA")
    print("=" * 90)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,

)


# ============================================================
# LOAD ONE 4-BIT MEDGEMMA COPY ON EACH DDP GPU
# ============================================================

print(
    f"[Rank {global_rank}] Loading MedGemma-27B "
    f"on local GPU {local_rank}...",
    flush=True,
)

model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_PATH),
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
    device_map={"": local_rank},
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    attn_implementation="eager",
)

model.config.use_cache = False
model.config.pad_token_id = tokenizer.pad_token_id

# Gradient checkpointing is enabled to reduce memory consumption.
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)

if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()


# ============================================================
# VERIFY LORA TARGET MODULES EXIST
# ============================================================

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

available_module_suffixes = {
    name.split(".")[-1]
    for name, _ in model.named_modules()
}

missing_targets = [
    name
    for name in LORA_TARGET_MODULES
    if name not in available_module_suffixes
]

if missing_targets:
    raise RuntimeError(
        "Missing MedGemma LoRA target modules: "
        f"{missing_targets}"
    )

if accelerator.is_main_process:
    print("Verified LoRA targets:", LORA_TARGET_MODULES)


# ============================================================
# LORA CONFIGURATION
# ============================================================

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.10,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=LORA_TARGET_MODULES,
)
# ============================================================
# TRAINING ARGUMENTS
# ============================================================

training_arguments = TrainingArguments(
    output_dir=str(OUTPUT_DIR),

    num_train_epochs=MAXIMUM_EPOCHS,

    per_device_train_batch_size=(
        PER_DEVICE_TRAIN_BATCH_SIZE
    ),

    per_device_eval_batch_size=(
        PER_DEVICE_EVAL_BATCH_SIZE
    ),

    gradient_accumulation_steps=(
        GRADIENT_ACCUMULATION_STEPS
    ),

    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type="cosine",

    optim="adamw_torch",

    bf16=True,
    fp16=False,
    tf32=True,

    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={
        "use_reentrant": True,
    },

    max_grad_norm=1.0,

    # Evaluate and save once after every complete epoch.
    eval_strategy="epoch",
    save_strategy="epoch",

    save_total_limit=2,

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    logging_strategy="steps",
    logging_steps=10,

    report_to="none",

    seed=RANDOM_SEED,
    data_seed=RANDOM_SEED,

    dataloader_num_workers=0,
    remove_unused_columns=True,

    # Faster/cleaner DDP for LoRA training.
    ddp_find_unused_parameters=False,
)


# ============================================================
# CREATE TRAINER
# ============================================================

accelerator.print("\n" + "=" * 90)
accelerator.print(
    "CREATING MEDGEMMA-27B TOPOGRAPHY "
    "4-GPU DDP QLORA TRAINER"
)
accelerator.print("=" * 90)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,

    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],

    peft_config=lora_config,

    dataset_text_field="text",
    max_seq_length=MAX_SEQUENCE_LENGTH,
    packing=False,

    args=training_arguments,
)

# ============================================================
# PARAMETER SUMMARY
# ============================================================

trainable_parameters = sum(
    parameter.numel()
    for parameter in trainer.model.parameters()
    if parameter.requires_grad
)

loaded_parameters = sum(
    parameter.numel()
    for parameter in trainer.model.parameters()
)

trainable_percentage = (
    100
    * trainable_parameters
    / loaded_parameters
)

if accelerator.is_main_process:
    print("\n" + "=" * 90)
    print("PARAMETER SUMMARY")
    print("=" * 90)

    print(
        "Trainable parameters:",
        f"{trainable_parameters:,}",
    )

    print(
        "Loaded parameters:",
        f"{loaded_parameters:,}",
    )

    print(
        "Trainable percentage:",
        f"{trainable_percentage:.4f}%",
    )

    print("\nOptimization configuration:")
    print("Maximum epochs:", MAXIMUM_EPOCHS)

    print(
        "Effective batch size:",
        EFFECTIVE_BATCH_SIZE,
    )

    print("Learning rate:", LEARNING_RATE)

    print(
        "Maximum sequence length:",
        MAX_SEQUENCE_LENGTH,
    )


# ============================================================
# TRAIN
# ============================================================

accelerator.print("\n" + "=" * 90)
accelerator.print(
    "STARTING MEDGEMMA-27B TOPOGRAPHY "
    "4-GPU DDP QLORA TRAINING"
)
accelerator.print("=" * 90)

training_result = trainer.train()

accelerator.wait_for_everyone()


# ============================================================
# TRAINING RESULT
# ============================================================

trainer_state = trainer.state

actual_epochs_completed = (
    float(trainer_state.epoch)
    if trainer_state.epoch is not None
    else None
)

best_checkpoint = trainer_state.best_model_checkpoint
best_validation_loss = trainer_state.best_metric
global_steps = trainer_state.global_step

if accelerator.is_main_process:
    print("\n" + "=" * 90)
    print("MODEL-SELECTION RESULT")
    print("=" * 90)

    print(
        "Actual epochs completed:",
        actual_epochs_completed,
    )

    print(
        "Global optimization steps:",
        global_steps,
    )

    print(
        "Best checkpoint:",
        best_checkpoint,
    )

    print(
        "Best validation loss:",
        best_validation_loss,
    )


# ============================================================
# SAVE RESTORED BEST ADAPTER
# ============================================================

accelerator.print("\n" + "=" * 90)
accelerator.print(
    "SAVING BEST MEDGEMMA-27B "
    "TOPOGRAPHY ADAPTER"
)
accelerator.print("=" * 90)

trainer.save_model(
    str(BEST_ADAPTER_DIR)
)

if accelerator.is_main_process:
    tokenizer.save_pretrained(
        str(BEST_ADAPTER_DIR)
    )

trainer.save_state()

accelerator.wait_for_everyone()


# ============================================================
# SAVE TRAINING METRICS
# ============================================================

if accelerator.is_main_process:
    trainer.log_metrics(
        "train",
        training_result.metrics,
    )

    trainer.save_metrics(
        "train",
        training_result.metrics,
    )


# ============================================================
# EVALUATE RESTORED BEST MODEL
# ============================================================

accelerator.print("\n" + "=" * 90)
accelerator.print("EVALUATING RESTORED BEST MODEL")
accelerator.print("=" * 90)

validation_metrics = trainer.evaluate()

accelerator.wait_for_everyone()

if accelerator.is_main_process:
    trainer.log_metrics(
        "validation",
        validation_metrics,
    )

    trainer.save_metrics(
        "validation",
        validation_metrics,
    )


# ============================================================
# SAVE TRAINING SUMMARY
# ============================================================

if accelerator.is_main_process:
    summary = {
        "model": "google/medgemma-27b-text-it",
        "task": "topography",
        "training_method": (
            "4-bit NF4 QLoRA with 4-GPU DDP"
        ),
        "distributed_processes": world_size,
        "gpu_count": torch.cuda.device_count(),
        "maximum_epochs": MAXIMUM_EPOCHS,
        "actual_epochs_completed": (
            actual_epochs_completed
        ),
        "global_optimization_steps": global_steps,
        "best_checkpoint": best_checkpoint,
        "best_validation_loss": (
            best_validation_loss
        ),
        "per_device_train_batch_size": (
            PER_DEVICE_TRAIN_BATCH_SIZE
        ),
        "per_device_eval_batch_size": (
            PER_DEVICE_EVAL_BATCH_SIZE
        ),
        "gradient_accumulation_steps": (
            GRADIENT_ACCUMULATION_STEPS
        ),
        "effective_batch_size": (
            EFFECTIVE_BATCH_SIZE
        ),
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "learning_rate_scheduler": "cosine",
        "optimizer": "AdamW",
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.10,
        "lora_target_modules": LORA_TARGET_MODULES,
        "quantization_type": "4-bit NF4",
        "double_quantization": True,
        "compute_dtype": "bfloat16",
        "maximum_sequence_length": (
            MAX_SEQUENCE_LENGTH
        ),
        "trainable_parameters": (
            trainable_parameters
        ),
        "loaded_parameters": loaded_parameters,
        "trainable_percentage": (
            trainable_percentage
        ),
        "training_metrics": (
            training_result.metrics
        ),
        "restored_best_model_validation_metrics": (
            validation_metrics
        ),
        "best_adapter_directory": (
            str(BEST_ADAPTER_DIR)
        ),
    }

    with TRAINING_SUMMARY_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            default=str,
        )


# ============================================================
# COMPLETE
# ============================================================

accelerator.wait_for_everyone()

if accelerator.is_main_process:
    print("\n" + "=" * 90)
    print(
        "MEDGEMMA-27B TOPOGRAPHY "
        "TRAINING COMPLETED"
    )
    print("=" * 90)

    print(
        "Actual epochs completed:",
        actual_epochs_completed,
    )

    print(
        "Best validation loss:",
        best_validation_loss,
    )

    print(
        "\nBest adapter saved at:\n",
        BEST_ADAPTER_DIR,
    )

    print(
        "\nTraining summary saved at:\n",
        TRAINING_SUMMARY_FILE,
    )

    print(
        "\nRestored-best-model validation metrics:"
    )

    for metric_name, metric_value in (
        validation_metrics.items()
    ):
        print(
            f"{metric_name}: "
            f"{metric_value}"
        )
