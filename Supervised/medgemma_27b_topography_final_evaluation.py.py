from __future__ import annotations

from pathlib import Path
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
    "medgemma-27b-text-it"
)

ADAPTER_PATH = Path(
    "/workspace/Reviewer_comment/checkpoints/"
    "medgemma_27b_topography_qlora_full/"
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
    "medgemma_27b_topography_qlora_final"
)

PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "medgemma_27b_topography_three_mode_predictions.csv"
)


SUMMARY_FILE = (
    OUTPUT_DIR
    / "medgemma_27b_topography_three_mode_summary_metrics.csv"
)

PRIMARY_COMPARISON_FILE = (
    OUTPUT_DIR
    / "medgemma_27b_topography_primary_all_test_comparison.csv"
)


RUN_SUMMARY_FILE = (
    OUTPUT_DIR
    / "medgemma_27b_topography_evaluation_run_summary.json"
)

# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

MAX_INPUT_LENGTH = 512

# Unconstrained output can contain explanatory text.
RAW_MAX_NEW_TOKENS = 32

# A five-digit code normally requires fewer tokens, but this
# leaves sufficient space for tokenization plus EOS.
PCD_MAX_NEW_TOKENS = 12

SAVE_EVERY = 10

CODE_PATTERN = re.compile(r"(?<!\d)(\d{5})(?!\d)")

INVALID_LABEL = "__INVALID__"

MAX_MEMORY = {
    0: "37GiB",
    1: "37GiB",
    2: "37GiB",
    3: "37GiB",
    "cpu": "200GiB",
}


if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

if torch.cuda.device_count() < 1:
    raise RuntimeError(
        "This evaluation requires at least one visible CUDA GPU."
    )


# ============================================================
# BASIC SETUP
# ============================================================

set_seed(RANDOM_SEED)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

for required_path in [
    MODEL_PATH,
    ADAPTER_PATH,
    TRAIN_FILE,
    TEST_FILE,
]:
    if not required_path.exists():
        raise FileNotFoundError(
            f"Required path does not exist:\n{required_path}"
        )


# ============================================================
# JSONL HELPERS
# ============================================================

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {path}, line {line_number}"
                ) from error

            records.append(record)

    return records


def normalize_gold_code(value: Any) -> str:
    text = str(value).strip()

    exact_match = re.fullmatch(r"\d{5}", text)

    if exact_match is not None:
        return exact_match.group(0)

    embedded_match = CODE_PATTERN.search(text)

    if embedded_match is not None:
        return embedded_match.group(1)

    raise ValueError(
        f"Could not find a five-digit gold code in: {value!r}"
    )


# ============================================================
# LOAD TRAINING AND TEST METADATA
# ============================================================

print("=" * 100)
print("LOADING TOPOGRAPHY TRAINING AND TEST DATA")
print("=" * 100)

train_records = read_jsonl(TRAIN_FILE)
test_records = read_jsonl(TEST_FILE)

train_codes = sorted(
    {
        normalize_gold_code(record["output"])
        for record in train_records
    }
)

train_code_set = set(train_codes)

print("Training reports:", f"{len(train_records):,}")
print("Test reports:", f"{len(test_records):,}")
print("Closed-set training codes:", f"{len(train_codes):,}")

if len(test_records) != 4272:
    print(
        "WARNING: Expected 4,272 test reports, but found",
        len(test_records),
    )


# ============================================================
# LOAD TOKENIZER
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
tokenizer.padding_side = "left"

# Preserve the diagnostic conclusion and TOPOGRAPHY_CODE
# response cue if a report exceeds the inference context limit.
tokenizer.truncation_side = "left"

print("Vocabulary size:", len(tokenizer))
print("EOS token:", tokenizer.eos_token)
print("PAD token:", tokenizer.pad_token)


# ============================================================
# LOAD QUANTIZED BASE MODEL
# ============================================================

print("\n" + "=" * 100)
print("LOADING MEDGEMMA-27B BASE MODEL IN 4-BIT")
print("=" * 100)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_PATH),
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    attn_implementation="eager",
)

base_model.config.use_cache = True
base_model.config.pad_token_id = tokenizer.pad_token_id


# ============================================================
# LOAD TRAINED ADAPTER
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

print("Input/embedding device:", input_device)

device_map = getattr(
    base_model,
    "hf_device_map",
    {},
)

device_counts: dict[str, int] = {}

for _, device in device_map.items():
    device_name = str(device)
    device_counts[device_name] = (
        device_counts.get(device_name, 0) + 1
    )

print("\nDevice-map summary:")

for device_name, count in sorted(device_counts.items()):
    print(
        f"  Device {device_name}: "
        f"{count} mapped modules"
    )

cpu_offloaded = sum(
    1
    for device in device_map.values()
    if str(device) == "cpu"
)

disk_offloaded = sum(
    1
    for device in device_map.values()
    if str(device) == "disk"
)

print("CPU-offloaded modules:", cpu_offloaded)
print("Disk-offloaded modules:", disk_offloaded)


# ============================================================
# PROMPT
# ============================================================

def build_prompt(record: dict[str, Any]) -> str:
    instruction = str(
        record["instruction"]
    ).strip()

    report = str(
        record["input"]
    ).strip()

    return (
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Pathology report:\n"
        f"{report}\n\n"
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
    after stripping surrounding whitespace, the generated suffix
    must contain exactly one five-digit code and nothing else.
    """

    cleaned = generated_text.strip()

    if re.fullmatch(r"\d{5}", cleaned):
        return cleaned, True

    return None, False


def relaxed_first_five_digit_prediction(
    generated_text: str,
) -> tuple[str | None, bool]:
    """
    Relaxed First-Five-Digit Extraction:
    extract the first standalone five-digit sequence appearing
    anywhere in the unconstrained generated suffix.
    """

    match = CODE_PATTERN.search(generated_text)

    if match is None:
        return None, False

    return match.group(1), True


# ============================================================
# BUILD PREFIX TRIE FOR PCD
# ============================================================

class TokenTrie:
    TERMINAL = "__terminal__"

    def __init__(self) -> None:
        self.root: dict[Any, Any] = {}

    def insert(self, token_ids: list[int]) -> None:
        node = self.root

        for token_id in token_ids:
            node = node.setdefault(
                int(token_id),
                {},
            )

        node[self.TERMINAL] = True

    def allowed_next(
        self,
        prefix: list[int],
        eos_token_id: int,
    ) -> list[int]:
        node = self.root

        for token_id in prefix:
            token_id = int(token_id)

            if token_id not in node:
                # Defensive fallback. This should not normally occur.
                return [eos_token_id]

            node = node[token_id]

        allowed = [
            int(token_id)
            for token_id in node.keys()
            if token_id != self.TERMINAL
        ]

        if self.TERMINAL in node:
            allowed.append(eos_token_id)

        if not allowed:
            allowed = [eos_token_id]

        return sorted(set(allowed))


code_trie = TokenTrie()

# The prompt ends directly after the colon. Each candidate begins
# with one space, matching the format used during training.
code_token_sequences: dict[str, list[int]] = {}

for code in train_codes:
    token_ids = tokenizer.encode(
        f" {code}",
        add_special_tokens=False,
    )

    if not token_ids:
        raise RuntimeError(
            f"Tokenizer produced no tokens for code {code}"
        )

    code_token_sequences[code] = token_ids
    code_trie.insert(token_ids)

maximum_code_token_length = max(
    len(token_ids)
    for token_ids in code_token_sequences.values()
)

print(
    "\nMaximum tokenized closed-set code length:",
    maximum_code_token_length,
)


# ============================================================
# GENERATION FUNCTIONS
# ============================================================

@torch.inference_mode()
def generate_unconstrained(
    prompt: str,
) -> str:
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
        add_special_tokens=True,
    )

    input_ids = encoded["input_ids"].to(input_device)
    attention_mask = encoded["attention_mask"].to(input_device)

    prompt_length = input_ids.shape[1]

    generated_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=RAW_MAX_NEW_TOKENS,
        do_sample=False,
        num_beams=1,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    new_token_ids = generated_ids[
        0,
        prompt_length:,
    ]

    return tokenizer.decode(
        new_token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


@torch.inference_mode()
def generate_prefix_constrained_decoding(
    prompt: str,
) -> str:
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
        add_special_tokens=True,
    )

    input_ids = encoded["input_ids"].to(input_device)
    attention_mask = encoded["attention_mask"].to(input_device)

    prompt_length = input_ids.shape[1]

    def prefix_allowed_tokens_fn(
        batch_id: int,
        current_input_ids: torch.Tensor,
    ) -> list[int]:
        del batch_id

        generated_prefix = current_input_ids[
            prompt_length:
        ].tolist()

        return code_trie.allowed_next(
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
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    new_token_ids = generated_ids[
        0,
        prompt_length:,
    ]

    decoded = tokenizer.decode(
        new_token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()

    match = CODE_PATTERN.search(decoded)

    if match is None:
        raise RuntimeError(
            "PCD failed to produce a five-digit code. "
            f"Decoded suffix: {decoded!r}"
        )

    predicted_code = match.group(1)

    if predicted_code not in train_code_set:
        raise RuntimeError(
            "PCD produced a code outside the training closed set: "
            f"{predicted_code}"
        )

    return predicted_code


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

    if "row_index" not in existing_dataframe.columns:
        raise RuntimeError(
            "Existing predictions file does not contain row_index."
        )

    existing_dataframe["row_index"] = (
        existing_dataframe["row_index"].astype(int)
    )

    existing_rows = (
        existing_dataframe.to_dict("records")
    )

    completed_indices = set(
        existing_dataframe["row_index"].tolist()
    )

    print(
        "\nResuming existing evaluation:",
        f"{len(completed_indices):,}",
        "reports already completed.",
    )


# ============================================================
# EVALUATION LOOP
# ============================================================

print("\n" + "=" * 100)
print("STARTING MEDGEMMA-27B TOPOGRAPHY THREE-MODE EVALUATION")
print("=" * 100)

start_time = time.time()
rows = existing_rows.copy()

for row_index, record in enumerate(test_records):
    if row_index in completed_indices:
        continue

    gold_code = normalize_gold_code(
        record["output"]
    )

    seen_in_training = (
        gold_code in train_code_set
    )

    prompt = build_prompt(record)

    raw_generation = generate_unconstrained(
        prompt
    )

    srg_prediction, srg_valid = (
        strict_raw_prediction(raw_generation)
    )

    rfe_prediction, rfe_valid = (
        relaxed_first_five_digit_prediction(
            raw_generation
        )
    )

    pcd_prediction = (
        generate_prefix_constrained_decoding(prompt)
    )

    record_id = record.get(
        "id",
        row_index,
    )

    row = {
        "row_index": row_index,
        "id": record_id,
        "gold_code": gold_code,
        "seen_in_training": seen_in_training,

        "raw_generation": raw_generation,

        "strict_raw_prediction": (
            srg_prediction
            if srg_prediction is not None
            else ""
        ),
        "strict_raw_valid": srg_valid,
        "strict_raw_correct": (
            srg_prediction == gold_code
            if srg_prediction is not None
            else False
        ),

        "relaxed_first_five_digits_prediction": (
            rfe_prediction
            if rfe_prediction is not None
            else ""
        ),
        "relaxed_first_five_digits_valid": rfe_valid,
        "relaxed_first_five_digits_correct": (
            rfe_prediction == gold_code
            if rfe_prediction is not None
            else False
        ),

        "prefix_constrained_decoding_prediction": (
            pcd_prediction
        ),
        "prefix_constrained_decoding_valid": True,
        "prefix_constrained_decoding_correct": (
            pcd_prediction == gold_code
        ),
    }

    rows.append(row)

    completed_count = len(rows)

    if (
        completed_count % SAVE_EVERY == 0
        or completed_count == len(test_records)
    ):
        output_dataframe = pd.DataFrame(rows)

        output_dataframe = output_dataframe.sort_values(
            "row_index"
        ).reset_index(drop=True)

        output_dataframe.to_csv(
            PREDICTIONS_FILE,
            index=False,
        )

        elapsed_seconds = time.time() - start_time

        newly_completed = (
            completed_count - len(existing_rows)
        )

        average_seconds = (
            elapsed_seconds / newly_completed
            if newly_completed > 0
            else 0.0
        )

        remaining = (
            len(test_records) - completed_count
        )

        estimated_remaining_hours = (
            remaining
            * average_seconds
            / 3600
        )

        print(
            f"Completed {completed_count:,}/"
            f"{len(test_records):,} | "
            f"latest row={row_index} | "
            f"gold={gold_code} | "
            f"SRG={srg_prediction} | "
            f"RFE={rfe_prediction} | "
            f"PCD={pcd_prediction} | "
            f"estimated remaining="
            f"{estimated_remaining_hours:.2f} h",
            flush=True,
        )


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

predictions["seen_in_training"] = (
    predictions["seen_in_training"]
    .astype(str)
    .str.lower()
    .map(
        {
            "true": True,
            "false": False,
        }
    )
)

if len(predictions) != len(test_records):
    raise RuntimeError(
        "Prediction count does not match test count: "
        f"{len(predictions)} versus {len(test_records)}"
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

    valid_mask = (
        dataframe[valid_column]
        .astype(str)
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
            }
        )
        .fillna(False)
    )

    y_pred: list[str] = []

    for prediction, is_valid in zip(
        dataframe[prediction_column],
        valid_mask,
    ):
        if not bool(is_valid) or pd.isna(prediction):
            y_pred.append(INVALID_LABEL)
        else:
            prediction_text = str(prediction).strip()

            if not prediction_text:
                y_pred.append(INVALID_LABEL)
            else:
                y_pred.append(prediction_text)

    # Restrict class-averaged metrics to genuine ground-truth
    # topography classes. INVALID_LABEL remains in y_pred so
    # malformed generations still count as incorrect predictions,
    # but it is not added as an artificial class in macro-F1.
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

    return {
        "n": int(len(dataframe)),
        "accuracy": float(accuracy),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "invalid_predictions": invalid_predictions,
        "invalid_prediction_rate": float(
            invalid_predictions / len(dataframe)
        ),
    }


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
        predictions["seen_in_training"] == True
    ],
    "unseen_classes_only": predictions[
        predictions["seen_in_training"] == False
    ],
}

summary_rows: list[dict[str, Any]] = []

for mode_name, mode_columns in mode_configuration.items():
    for subset_name, subset_dataframe in (
        subset_configuration.items()
    ):
        metric_values = calculate_metrics(
            subset_dataframe,
            prediction_column=(
                mode_columns["prediction_column"]
            ),
            valid_column=(
                mode_columns["valid_column"]
            ),
        )

        summary_rows.append(
            {
                "mode": mode_name,
                "subset": subset_name,
                **metric_values,
            }
        )

summary_dataframe = pd.DataFrame(
    summary_rows
)

summary_dataframe.to_csv(
    SUMMARY_FILE,
    index=False,
)

primary_comparison = summary_dataframe[
    summary_dataframe["subset"]
    == "all_test_reports"
].copy()

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
    240,
)

print("\n" + "=" * 120)
print("MEDGEMMA-27B TOPOGRAPHY THREE-MODE SUMMARY METRICS")
print("=" * 120)

print(
    summary_dataframe.to_string(
        index=False,
    )
)

print("\n" + "=" * 120)
print("PRIMARY ALL-TEST COMPARISON")
print("=" * 120)

print(
    primary_comparison.to_string(
        index=False,
    )
)


# ============================================================
# SAVE RUN SUMMARY
# ============================================================

total_runtime_seconds = (
    time.time() - start_time
)

run_summary = {
    "model": "MedGemma-27B",
    "adapter": str(ADAPTER_PATH),
    "task": "topography",
    "test_reports": len(test_records),
    "training_closed_set_codes": len(train_codes),
    "seen_test_reports": int(
        predictions["seen_in_training"].sum()
    ),
    "unseen_test_reports": int(
        (~predictions["seen_in_training"]).sum()
    ),
    "evaluation_modes": [
        "strict_raw_generation",
        "relaxed_first_five_digits",
        "prefix_constrained_decoding",
    ],
    "pcd_candidate_source": (
        "unique topography codes in topography_train.jsonl"
    ),
    "predictions_file": str(PREDICTIONS_FILE),
    "summary_metrics_file": str(SUMMARY_FILE),
    "primary_comparison_file": str(
        PRIMARY_COMPARISON_FILE
    ),
    "runtime_seconds_for_current_invocation": (
        total_runtime_seconds
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

print("\nSaved predictions:")
print(PREDICTIONS_FILE)

print("\nSaved summary metrics:")
print(SUMMARY_FILE)

print("\nSaved primary comparison:")
print(PRIMARY_COMPARISON_FILE)

print("\nSaved run summary:")
print(RUN_SUMMARY_FILE)

print("\nEvaluation completed successfully.")
