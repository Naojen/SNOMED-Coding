from __future__ import annotations

from pathlib import Path
import gc
import json
import re
import time
from typing import Any

import pandas as pd
import torch
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = Path(
    "/workspace/Reviewer_comment/models/"
    "Meta-Llama-3-70B-Instruct"
)

ADAPTER_PATH = Path(
    "/workspace/Reviewer_comment/checkpoints/"
    "llama3_70b_topography_qlora_early_stopping/"
    "best_adapter"
)

TRAIN_FILE = Path(
    "/workspace/Reviewer_comment/supervised_data/splits/"
    "topography_train.jsonl"
)

TEST_FILE = Path(
    "/workspace/Reviewer_comment/supervised_data/splits/"
    "topography_test.jsonl"
)

OUTPUT_DIR = Path(
    "/workspace/Reviewer_comment/evaluation/"
    "llama3_70b_topography_qlora_final"
)

PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "llama3_70b_topography_three_mode_predictions.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "llama3_70b_topography_three_mode_summary_metrics.csv"
)

PRIMARY_COMPARISON_FILE = (
    OUTPUT_DIR
    / "llama3_70b_topography_primary_all_test_comparison.csv"
)

RUN_SUMMARY_FILE = (
    OUTPUT_DIR
    / "llama3_70b_topography_evaluation_run_summary.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

MAX_INPUT_LENGTH = 512

# Begin with two reports per generation batch.
# If CUDA runs out of memory, the script automatically retries
# each report individually.
BATCH_SIZE = 2

RAW_MAX_NEW_TOKENS = 16

PCD_MAX_NEW_TOKENS = 10

SAVE_EVERY_REPORTS = 20

EXPECTED_TEST_REPORTS = 4272

CODE_PATTERN = re.compile(
    r"(?<!\d)(\d{5})(?!\d)"
)

INVALID_LABEL = "__INVALID__"

# The DGX should expose all eight A100 GPUs as
# local cuda:0 through cuda:7.
MAX_MEMORY = {
    0: "37GiB",
    1: "37GiB",
    2: "37GiB",
    3: "37GiB",
    4: "37GiB",
    5: "37GiB",
    6: "37GiB",
    7: "37GiB",
    "cpu": "200GiB",
}


# ============================================================
# ENVIRONMENT
# ============================================================

set_seed(RANDOM_SEED)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available."
    )

if torch.cuda.device_count() != 8:
    raise RuntimeError(
        "This evaluation expects exactly eight visible GPUs, "
        f"but PyTorch sees {torch.cuda.device_count()}."
    )

print("=" * 100)
print("GPU CONFIGURATION")
print("=" * 100)

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
# CHECK REQUIRED PATHS
# ============================================================

for required_path in [
    MODEL_PATH,
    ADAPTER_PATH,
    TRAIN_FILE,
    TEST_FILE,
]:
    if not required_path.exists():
        raise FileNotFoundError(
            f"Required path does not exist:\n"
            f"{required_path}"
        )


# ============================================================
# JSONL HELPERS
# ============================================================

def read_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path}, "
                    f"line {line_number}"
                ) from error

            records.append(record)

    return records


def normalize_gold_code(
    value: Any,
) -> str:
    text = str(value).strip()

    exact_match = re.fullmatch(
        r"\d{5}",
        text,
    )

    if exact_match is not None:
        return exact_match.group(0)

    embedded_match = CODE_PATTERN.search(text)

    if embedded_match is not None:
        return embedded_match.group(1)

    raise ValueError(
        "Could not identify a five-digit topography "
        f"code in gold value: {value!r}"
    )


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 100)
print("LOADING TOPOGRAPHY DATA")
print("=" * 100)

train_records = read_jsonl(TRAIN_FILE)
test_records = read_jsonl(TEST_FILE)

train_codes = sorted(
    {
        normalize_gold_code(
            record["output"]
        )
        for record in train_records
    }
)

train_code_set = set(train_codes)

print(
    "Training reports:",
    f"{len(train_records):,}",
)

print(
    "Test reports:",
    f"{len(test_records):,}",
)

print(
    "Unique closed-set topography codes:",
    f"{len(train_codes):,}",
)

if len(test_records) != EXPECTED_TEST_REPORTS:
    print(
        "WARNING: Expected",
        EXPECTED_TEST_REPORTS,
        "test reports but found",
        len(test_records),
    )


# ============================================================
# TOKENIZER
# ============================================================

print("\n" + "=" * 100)
print("LOADING TOKENIZER")
print("=" * 100)

tokenizer = AutoTokenizer.from_pretrained(
    str(ADAPTER_PATH),
    use_fast=True,
    trust_remote_code=True,
)

if tokenizer.eos_token_id is None:
    raise RuntimeError(
        "Tokenizer does not define an EOS token."
    )

tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# Left padding is required for batched decoder-only generation.
tokenizer.padding_side = "left"

# Preserve the end of long prompts, including the diagnostic
# conclusion and the TOPOGRAPHY_CODE response cue.
tokenizer.truncation_side = "left"

print("Vocabulary size:", len(tokenizer))
print("EOS token:", tokenizer.eos_token)
print("PAD token:", tokenizer.pad_token)


# ============================================================
# LOAD BASE MODEL
# ============================================================

print("\n" + "=" * 100)
print("LOADING LLAMA-3-70B-INSTRUCT BASE MODEL IN 4-BIT")
print("=" * 100)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_PATH),

    quantization_config=(
        quantization_config
    ),

    torch_dtype=torch.bfloat16,

    # One model divided across all eight visible GPUs.
    device_map="balanced",

    max_memory=MAX_MEMORY,

    low_cpu_mem_usage=True,

    trust_remote_code=True,
)

base_model.config.use_cache = True
base_model.config.pad_token_id = (
    tokenizer.pad_token_id
)


# ============================================================
# LOAD TRAINED TOPOGRAPHY ADAPTER
# ============================================================

print("\n" + "=" * 100)
print("LOADING BEST TOPOGRAPHY ADAPTER")
print("=" * 100)

model = PeftModel.from_pretrained(
    base_model,
    str(ADAPTER_PATH),
    is_trainable=False,
)

model.eval()

input_device = (
    model.get_input_embeddings()
    .weight
    .device
)

print("Input embedding device:", input_device)

device_map = getattr(
    base_model,
    "hf_device_map",
    {},
)

device_counts: dict[str, int] = {}

for device in device_map.values():
    device_name = str(device)

    device_counts[device_name] = (
        device_counts.get(device_name, 0) + 1
    )

print("\nDevice-map summary:")

for device_name, module_count in sorted(
    device_counts.items()
):
    print(
        f"  Device {device_name}: "
        f"{module_count} mapped modules"
    )

cpu_offloaded_modules = sum(
    1
    for device in device_map.values()
    if str(device) == "cpu"
)

disk_offloaded_modules = sum(
    1
    for device in device_map.values()
    if str(device) == "disk"
)

print(
    "CPU-offloaded modules:",
    cpu_offloaded_modules,
)

print(
    "Disk-offloaded modules:",
    disk_offloaded_modules,
)

if disk_offloaded_modules > 0:
    raise RuntimeError(
        "Disk-offloaded modules were detected. "
        "Evaluation would be extremely slow."
    )


# ============================================================
# PROMPT
# ============================================================

def build_prompt(
    record: dict[str, Any],
) -> str:
    instruction = str(
        record["instruction"]
    ).strip()

    pathology_report = str(
        record["input"]
    ).strip()

    return (
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Pathology report:\n"
        f"{pathology_report}\n\n"
        "### Response:\n"
        "TOPOGRAPHY_CODE:"
    )


# ============================================================
# STRICT AND RELAXED PARSING
# ============================================================

def strict_raw_prediction(
    generated_text: str,
) -> tuple[str | None, bool]:
    """
    Strict Raw Generation:

    The generated suffix must contain exactly one standalone
    five-digit code after surrounding whitespace is removed.
    Any explanatory text, label, punctuation or malformed code
    makes the prediction invalid.
    """

    cleaned = generated_text.strip()

    if re.fullmatch(
        r"\d{5}",
        cleaned,
    ):
        return cleaned, True

    return None, False


def relaxed_first_five_digit_prediction(
    generated_text: str,
) -> tuple[str | None, bool]:
    """
    Relaxed First-Five-Digit Extraction:

    Retains unconstrained generation and extracts the first
    standalone five-digit code appearing in the response.
    """

    match = CODE_PATTERN.search(
        generated_text
    )

    if match is None:
        return None, False

    return match.group(1), True


# ============================================================
# CLOSED-SET TOKEN TRIE
# ============================================================

class TokenTrie:
    TERMINAL_KEY = "__terminal__"

    def __init__(self) -> None:
        self.root: dict[Any, Any] = {}

    def insert(
        self,
        token_ids: list[int],
    ) -> None:
        node = self.root

        for token_id in token_ids:
            node = node.setdefault(
                int(token_id),
                {},
            )

        node[self.TERMINAL_KEY] = True

    def allowed_next_tokens(
        self,
        prefix_token_ids: list[int],
        eos_token_id: int,
    ) -> list[int]:
        node = self.root

        for token_id in prefix_token_ids:
            token_id = int(token_id)

            if token_id not in node:
                return [eos_token_id]

            node = node[token_id]

        allowed_tokens = [
            int(token_id)
            for token_id in node.keys()
            if token_id != self.TERMINAL_KEY
        ]

        if self.TERMINAL_KEY in node:
            allowed_tokens.append(
                eos_token_id
            )

        if not allowed_tokens:
            allowed_tokens = [
                eos_token_id
            ]

        return sorted(
            set(allowed_tokens)
        )


code_trie = TokenTrie()

code_token_sequences: dict[
    str,
    list[int],
] = {}

for code in train_codes:
    # The model was trained with a space between the label
    # and the five-digit output code.
    token_ids = tokenizer.encode(
        f" {code}",
        add_special_tokens=False,
    )

    if not token_ids:
        raise RuntimeError(
            f"No tokenizer IDs produced for code {code}"
        )

    code_token_sequences[code] = token_ids
    code_trie.insert(token_ids)

maximum_code_token_length = max(
    len(token_ids)
    for token_ids in code_token_sequences.values()
)

print(
    "\nMaximum token length among closed-set codes:",
    maximum_code_token_length,
)


# ============================================================
# CUDA CLEANUP
# ============================================================

def clear_cuda_cache() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# BATCHED UNCONSTRAINED GENERATION
# ============================================================

@torch.inference_mode()
def generate_unconstrained_batch(
    prompts: list[str],
) -> list[str]:
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
        add_special_tokens=True,
    )

    input_ids = encoded[
        "input_ids"
    ].to(input_device)

    attention_mask = encoded[
        "attention_mask"
    ].to(input_device)

    padded_prompt_length = (
        input_ids.shape[1]
    )

    generated_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,

        max_new_tokens=RAW_MAX_NEW_TOKENS,

        do_sample=False,
        num_beams=1,
        use_cache=True,

        pad_token_id=(
            tokenizer.pad_token_id
        ),

        eos_token_id=(
            tokenizer.eos_token_id
        ),
    )

    decoded_suffixes: list[str] = []

    for row_ids in generated_ids:
        new_token_ids = row_ids[
            padded_prompt_length:
        ]

        decoded_suffix = tokenizer.decode(
            new_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        decoded_suffixes.append(
            decoded_suffix
        )

    del encoded
    del input_ids
    del attention_mask
    del generated_ids

    return decoded_suffixes


# ============================================================
# BATCHED CONSTRAINED GENERATION
# ============================================================

@torch.inference_mode()
def generate_constrained_batch(
    prompts: list[str],
) -> list[str]:
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
        add_special_tokens=True,
    )

    input_ids = encoded[
        "input_ids"
    ].to(input_device)

    attention_mask = encoded[
        "attention_mask"
    ].to(input_device)

    padded_prompt_length = (
        input_ids.shape[1]
    )

    def prefix_allowed_tokens_fn(
        batch_id: int,
        current_input_ids: torch.Tensor,
    ) -> list[int]:
        del batch_id

        generated_prefix = (
            current_input_ids[
                padded_prompt_length:
            ]
            .detach()
            .cpu()
            .tolist()
        )

        return code_trie.allowed_next_tokens(
            generated_prefix,
            tokenizer.eos_token_id,
        )

    generated_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,

        max_new_tokens=max(
            PCD_MAX_NEW_TOKENS,
            maximum_code_token_length + 2,
        ),

        do_sample=False,
        num_beams=1,
        use_cache=True,

        prefix_allowed_tokens_fn=(
            prefix_allowed_tokens_fn
        ),

        pad_token_id=(
            tokenizer.pad_token_id
        ),

        eos_token_id=(
            tokenizer.eos_token_id
        ),
    )

    constrained_predictions: list[str] = []

    for row_ids in generated_ids:
        new_token_ids = row_ids[
            padded_prompt_length:
        ]

        decoded_suffix = tokenizer.decode(
            new_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

        match = CODE_PATTERN.search(
            decoded_suffix
        )

        if match is None:
            raise RuntimeError(
                "PCD failed to produce a five-digit code. "
                f"Decoded suffix: {decoded_suffix!r}"
            )

        predicted_code = match.group(1)

        if predicted_code not in train_code_set:
            raise RuntimeError(
                "PCD produced an out-of-set code: "
                f"{predicted_code}"
            )

        constrained_predictions.append(
            predicted_code
        )

    del encoded
    del input_ids
    del attention_mask
    del generated_ids

    return constrained_predictions


# ============================================================
# GENERATION WITH AUTOMATIC OOM FALLBACK
# ============================================================

def generate_modes_with_fallback(
    prompts: list[str],
) -> tuple[list[str], list[str]]:
    """
    Tries batched inference first.

    If batch size 2 causes CUDA OOM, the same reports are
    immediately retried one at a time.
    """

    try:
        raw_outputs = (
            generate_unconstrained_batch(
                prompts
            )
        )

        constrained_outputs = (
            generate_constrained_batch(
                prompts
            )
        )

        return (
            raw_outputs,
            constrained_outputs,
        )

    except torch.OutOfMemoryError:
        print(
            "\nCUDA OOM during batched generation. "
            "Retrying this batch one report at a time.",
            flush=True,
        )

        clear_cuda_cache()

        if len(prompts) == 1:
            raise

        all_raw_outputs: list[str] = []
        all_constrained_outputs: list[str] = []

        for prompt in prompts:
            raw_output = (
                generate_unconstrained_batch(
                    [prompt]
                )[0]
            )

            constrained_output = (
                generate_constrained_batch(
                    [prompt]
                )[0]
            )

            all_raw_outputs.append(
                raw_output
            )

            all_constrained_outputs.append(
                constrained_output
            )

            clear_cuda_cache()

        return (
            all_raw_outputs,
            all_constrained_outputs,
        )


# ============================================================
# RESUME SUPPORT
# ============================================================

existing_rows: list[dict[str, Any]] = []
completed_indices: set[int] = set()

if PREDICTIONS_FILE.exists():
    existing_dataframe = pd.read_csv(
        PREDICTIONS_FILE,
        dtype=str,
    )

    if "row_index" not in (
        existing_dataframe.columns
    ):
        raise RuntimeError(
            "Existing prediction file does not "
            "contain row_index."
        )

    existing_dataframe["row_index"] = (
        existing_dataframe[
            "row_index"
        ].astype(int)
    )

    existing_rows = (
        existing_dataframe.to_dict(
            "records"
        )
    )

    completed_indices = set(
        existing_dataframe[
            "row_index"
        ].tolist()
    )

    print(
        "\nResuming evaluation:",
        f"{len(completed_indices):,}",
        "test reports already completed.",
    )


# ============================================================
# PREPARE PENDING ROWS
# ============================================================

pending_indices = [
    row_index
    for row_index in range(
        len(test_records)
    )
    if row_index not in completed_indices
]

print("\n" + "=" * 100)
print("STARTING LLAMA-3-70B-INSTRUCT TOPOGRAPHY THREE-MODE EVALUATION")
print("=" * 100)

print(
    "Already completed:",
    f"{len(completed_indices):,}",
)

print(
    "Pending reports:",
    f"{len(pending_indices):,}",
)

print(
    "Configured batch size:",
    BATCH_SIZE,
)


# ============================================================
# EVALUATION LOOP
# ============================================================

start_time = time.time()

rows = existing_rows.copy()

newly_completed_count = 0

for batch_start in range(
    0,
    len(pending_indices),
    BATCH_SIZE,
):
    batch_indices = pending_indices[
        batch_start:
        batch_start + BATCH_SIZE
    ]

    batch_records = [
        test_records[row_index]
        for row_index in batch_indices
    ]

    batch_prompts = [
        build_prompt(record)
        for record in batch_records
    ]

    (
        raw_generations,
        pcd_predictions,
    ) = generate_modes_with_fallback(
        batch_prompts
    )

    for (
        row_index,
        record,
        raw_generation,
        pcd_prediction,
    ) in zip(
        batch_indices,
        batch_records,
        raw_generations,
        pcd_predictions,
    ):
        gold_code = normalize_gold_code(
            record["output"]
        )

        seen_in_training = (
            gold_code in train_code_set
        )

        (
            strict_prediction,
            strict_valid,
        ) = strict_raw_prediction(
            raw_generation
        )

        (
            relaxed_prediction,
            relaxed_valid,
        ) = relaxed_first_five_digit_prediction(
            raw_generation
        )

        record_id = record.get(
            "id",
            row_index,
        )

        row = {
            "row_index": row_index,
            "id": record_id,

            "gold_code": gold_code,

            "seen_in_training": (
                seen_in_training
            ),

            "raw_generation": (
                raw_generation
            ),

            "strict_raw_prediction": (
                strict_prediction
                if strict_prediction is not None
                else ""
            ),

            "strict_raw_valid": (
                strict_valid
            ),

            "strict_raw_correct": (
                strict_prediction == gold_code
                if strict_prediction is not None
                else False
            ),

            "relaxed_first_five_digits_prediction": (
                relaxed_prediction
                if relaxed_prediction is not None
                else ""
            ),

            "relaxed_first_five_digits_valid": (
                relaxed_valid
            ),

            "relaxed_first_five_digits_correct": (
                relaxed_prediction == gold_code
                if relaxed_prediction is not None
                else False
            ),

            "prefix_constrained_decoding_prediction": (
                pcd_prediction
            ),

            "prefix_constrained_decoding_valid": (
                True
            ),

            "prefix_constrained_decoding_correct": (
                pcd_prediction == gold_code
            ),
        }

        rows.append(row)
        newly_completed_count += 1

    total_completed_count = len(rows)

    should_save = (
        newly_completed_count
        % SAVE_EVERY_REPORTS == 0
        or total_completed_count
        == len(test_records)
    )

    if should_save:
        output_dataframe = pd.DataFrame(
            rows
        )

        output_dataframe = (
            output_dataframe
            .sort_values("row_index")
            .drop_duplicates(
                subset=["row_index"],
                keep="last",
            )
            .reset_index(drop=True)
        )

        output_dataframe.to_csv(
            PREDICTIONS_FILE,
            index=False,
        )

        elapsed_seconds = (
            time.time() - start_time
        )

        average_seconds_per_report = (
            elapsed_seconds
            / newly_completed_count
            if newly_completed_count > 0
            else 0.0
        )

        remaining_reports = (
            len(test_records)
            - len(output_dataframe)
        )

        estimated_remaining_hours = (
            remaining_reports
            * average_seconds_per_report
            / 3600
        )

        latest_row = output_dataframe.iloc[
            -1
        ]

        print(
            f"Completed "
            f"{len(output_dataframe):,}/"
            f"{len(test_records):,} | "
            f"latest index="
            f"{int(latest_row['row_index'])} | "
            f"gold={latest_row['gold_code']} | "
            f"SRG="
            f"{latest_row['strict_raw_prediction']} | "
            f"RFE="
            f"{latest_row['relaxed_first_five_digits_prediction']} | "
            f"PCD="
            f"{latest_row['prefix_constrained_decoding_prediction']} | "
            f"estimated remaining="
            f"{estimated_remaining_hours:.2f} h",
            flush=True,
        )

        clear_cuda_cache()


# ============================================================
# LOAD FINAL PREDICTIONS
# ============================================================

predictions = pd.read_csv(
    PREDICTIONS_FILE,
    dtype={
        "gold_code": str,

        "strict_raw_prediction": str,

        "relaxed_first_five_digits_prediction": str,

        "prefix_constrained_decoding_prediction": str,
    },
)

predictions = (
    predictions
    .sort_values("row_index")
    .drop_duplicates(
        subset=["row_index"],
        keep="last",
    )
    .reset_index(drop=True)
)

if len(predictions) != len(test_records):
    raise RuntimeError(
        "Prediction count does not match the test set: "
        f"{len(predictions)} predictions versus "
        f"{len(test_records)} test reports."
    )


# ============================================================
# NORMALIZE BOOLEAN COLUMNS
# ============================================================

def normalize_boolean_series(
    series: pd.Series,
) -> pd.Series:
    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        )
        .fillna(False)
        .astype(bool)
    )


predictions["seen_in_training"] = (
    normalize_boolean_series(
        predictions["seen_in_training"]
    )
)

predictions["strict_raw_valid"] = (
    normalize_boolean_series(
        predictions["strict_raw_valid"]
    )
)

predictions[
    "relaxed_first_five_digits_valid"
] = normalize_boolean_series(
    predictions[
        "relaxed_first_five_digits_valid"
    ]
)

predictions[
    "prefix_constrained_decoding_valid"
] = normalize_boolean_series(
    predictions[
        "prefix_constrained_decoding_valid"
    ]
)


# ============================================================
# METRIC CALCULATION
# ============================================================

def calculate_metrics(
    dataframe: pd.DataFrame,
    prediction_column: str,
    valid_column: str,
) -> dict[str, Any]:
    if dataframe.empty:
        return {
            "n": 0,
            "accuracy": 0.0,

            "weighted_precision": 0.0,
            "weighted_recall": 0.0,
            "weighted_f1": 0.0,

            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,

            "micro_precision": 0.0,
            "micro_recall": 0.0,
            "micro_f1": 0.0,

            "invalid_predictions": 0,
            "invalid_prediction_rate": 0.0,
        }

    y_true = (
        dataframe["gold_code"]
        .astype(str)
        .tolist()
    )

    valid_mask = normalize_boolean_series(
        dataframe[valid_column]
    )

    y_pred: list[str] = []

    for prediction, is_valid in zip(
        dataframe[prediction_column],
        valid_mask,
    ):
        if (
            not bool(is_valid)
            or pd.isna(prediction)
        ):
            y_pred.append(
                INVALID_LABEL
            )
            continue

        prediction_text = str(
            prediction
        ).strip()

        if not prediction_text:
            y_pred.append(
                INVALID_LABEL
            )
        else:
            y_pred.append(
                prediction_text
            )

    # Only genuine ground-truth SNOMED classes are included in
    # class-averaged metrics. INVALID_LABEL remains in y_pred, so
    # malformed generations are still counted as errors/false
    # negatives, but it is not treated as an artificial clinical
    # class in macro-F1.
    evaluation_labels = sorted(set(y_true))

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    (
        weighted_precision,
        weighted_recall,
        weighted_f1,
        _,
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=evaluation_labels,
        average="weighted",
        zero_division=0,
    )

    (
        macro_precision,
        macro_recall,
        macro_f1,
        _,
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=evaluation_labels,
        average="macro",
        zero_division=0,
    )

    (
        micro_precision,
        micro_recall,
        micro_f1,
        _,
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="micro",
        zero_division=0,
    )

    invalid_predictions = int(
        (~valid_mask).sum()
    )

    invalid_prediction_rate = (
        invalid_predictions
        / len(dataframe)
    )

    return {
        "n": int(len(dataframe)),

        "accuracy": float(accuracy),

        "weighted_precision": float(
            weighted_precision
        ),

        "weighted_recall": float(
            weighted_recall
        ),

        "weighted_f1": float(
            weighted_f1
        ),

        "macro_precision": float(
            macro_precision
        ),

        "macro_recall": float(
            macro_recall
        ),

        "macro_f1": float(
            macro_f1
        ),

        "micro_precision": float(
            micro_precision
        ),

        "micro_recall": float(
            micro_recall
        ),

        "micro_f1": float(
            micro_f1
        ),

        "invalid_predictions": (
            invalid_predictions
        ),

        "invalid_prediction_rate": float(
            invalid_prediction_rate
        ),
    }


# ============================================================
# MODE AND SUBSET CONFIGURATION
# ============================================================

mode_configuration = {
    "strict_raw_generation": {
        "prediction_column": (
            "strict_raw_prediction"
        ),

        "valid_column": (
            "strict_raw_valid"
        ),
    },

    "relaxed_first_five_digits": {
        "prediction_column": (
            "relaxed_first_five_digits_prediction"
        ),

        "valid_column": (
            "relaxed_first_five_digits_valid"
        ),
    },

    "prefix_constrained_decoding": {
        "prediction_column": (
            "prefix_constrained_decoding_prediction"
        ),

        "valid_column": (
            "prefix_constrained_decoding_valid"
        ),
    },
}

subset_configuration = {
    "all_test_reports": predictions,

    "seen_classes_only": predictions[
        predictions["seen_in_training"]
    ],

    "unseen_classes_only": predictions[
        ~predictions["seen_in_training"]
    ],
}


# ============================================================
# COMPUTE SUMMARY METRICS
# ============================================================

summary_rows: list[
    dict[str, Any]
] = []

for (
    mode_name,
    mode_columns,
) in mode_configuration.items():

    for (
        subset_name,
        subset_dataframe,
    ) in subset_configuration.items():

        metrics = calculate_metrics(
            subset_dataframe,

            prediction_column=(
                mode_columns[
                    "prediction_column"
                ]
            ),

            valid_column=(
                mode_columns[
                    "valid_column"
                ]
            ),
        )

        summary_rows.append(
            {
                "mode": mode_name,
                "subset": subset_name,
                **metrics,
            }
        )

summary_dataframe = pd.DataFrame(
    summary_rows
)

summary_dataframe.to_csv(
    SUMMARY_FILE,
    index=False,
)

primary_comparison = (
    summary_dataframe[
        summary_dataframe["subset"]
        == "all_test_reports"
    ]
    .copy()
    .reset_index(drop=True)
)

primary_comparison.to_csv(
    PRIMARY_COMPARISON_FILE,
    index=False,
)


# ============================================================
# PRINT RESULTS
# ============================================================

pd.set_option(
    "display.max_columns",
    None,
)

pd.set_option(
    "display.width",
    260,
)

print("\n" + "=" * 130)
print(
    "LLAMA-3-70B-INSTRUCT TOPOGRAPHY "
    "THREE-MODE SUMMARY METRICS"
)
print("=" * 130)

print(
    summary_dataframe.to_string(
        index=False,
    )
)

print("\n" + "=" * 130)
print("PRIMARY ALL-TEST COMPARISON")
print("=" * 130)

print(
    primary_comparison.to_string(
        index=False,
    )
)


# ============================================================
# SAVE RUN SUMMARY
# ============================================================

runtime_seconds = (
    time.time() - start_time
)

seen_test_reports = int(
    predictions[
        "seen_in_training"
    ].sum()
)

unseen_test_reports = int(
    (
        ~predictions[
            "seen_in_training"
        ]
    ).sum()
)

run_summary = {
    "model": "Meta-Llama-3-70B-Instruct",

    "adapter": str(
        ADAPTER_PATH
    ),

    "task": "topography",

    "test_reports": len(
        test_records
    ),

    "training_closed_set_codes": len(
        train_codes
    ),

    "seen_test_reports": (
        seen_test_reports
    ),

    "unseen_test_reports": (
        unseen_test_reports
    ),

    "evaluation_modes": [
        "strict_raw_generation",
        "relaxed_first_five_digits",
        "prefix_constrained_decoding",
    ],

    "pcd_candidate_source": (
        "unique topography codes in "
        "topography_train.jsonl"
    ),

    "configured_batch_size": (
        BATCH_SIZE
    ),

    "raw_max_new_tokens": (
        RAW_MAX_NEW_TOKENS
    ),

    "pcd_max_new_tokens": (
        PCD_MAX_NEW_TOKENS
    ),

    "visible_gpu_count": (
        torch.cuda.device_count()
    ),

    "predictions_file": str(
        PREDICTIONS_FILE
    ),

    "summary_metrics_file": str(
        SUMMARY_FILE
    ),

    "primary_comparison_file": str(
        PRIMARY_COMPARISON_FILE
    ),

    "runtime_seconds_for_current_invocation": (
        runtime_seconds
    ),
}

with RUN_SUMMARY_FILE.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        run_summary,
        file,
        indent=2,
    )


# ============================================================
# COMPLETE
# ============================================================

print("\nSaved predictions:")
print(PREDICTIONS_FILE)

print("\nSaved summary metrics:")
print(SUMMARY_FILE)

print("\nSaved primary comparison:")
print(PRIMARY_COMPARISON_FILE)

print("\nSaved run summary:")
print(RUN_SUMMARY_FILE)

print("\nTopography evaluation completed successfully.")
