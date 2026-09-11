
## Supervised Fine-Tuning and Evaluation

This repository provides scripts for supervised SNOMED morphology and topography coding using QLoRA. Training and evaluation use the same development and held-out evaluation partitions employed in the PRISM experiments.

### Training configuration

The supervised models are fine-tuned using 4-bit QLoRA with the following configuration:

* NF4 quantization with double quantization
* Bfloat16 computation
* LoRA rank: 16
* LoRA scaling factor: 32
* LoRA dropout: 0.10
* LoRA target modules: query, key, value, output, gate, up, and down projection layers
* Optimizer: AdamW
* Peak learning rate: `1.5e-5`
* Weight decay: `0.01`
* Warmup ratio: `0.05`
* Learning-rate schedule: cosine decay
* Global effective batch size: 32
* Maximum training duration: 15 epochs
* Model selection criterion: lowest validation loss

Gradient checkpointing and Fully Sharded Data Parallelism (FSDP) are used to reduce GPU-memory requirements. The example configuration uses four GPUs with a per-device batch size of 1 and eight gradient-accumulation steps:

```text
1 sample per GPU × 4 GPUs × 8 accumulation steps = 32
```

### Data format

Training and validation files must use JSON Lines format with the following fields:

```json
{
  "instruction": "Assign the appropriate SNOMED morphology code.",
  "input": "Pathology report text",
  "output": "81403"
}
```

Morphology and topography models must be trained separately using their corresponding dataset partitions.

### Running QLoRA training

Update the model, dataset, output, and configuration paths at the beginning of the training script before execution.

An example four-GPU FSDP launch command is:

```bash
cd /workspace/Reviewer_comment

accelerate launch \
  --config_file /workspace/Reviewer_comment/fsdp_qlora_70b_full.yaml \
  /workspace/Reviewer_comment/llama3_70b_morphology_qlora_full.py
```

For topography training, use the corresponding topography training script and replace the morphology dataset paths with:

```text
topography_train.jsonl
topography_validation.jsonl
```

The best checkpoint is selected according to validation loss and saved in the output directory defined in the training script.

### Evaluation protocols

The supervised models are evaluated under three inference protocols:

1. **Strict Raw Generation (SRG):** The unconstrained output is considered valid only when it contains exactly one standalone five-digit code. Outputs containing additional labels, explanations, punctuation, or malformed continuations are treated as incorrect.

2. **Relaxed First-Code Extraction (RFE):** Generation remains unconstrained, but the first standalone five-digit code is extracted from the generated response. A response containing no valid five-digit code is treated as incorrect.

3. **Prefix-Constrained Decoding (PCD):** During decoding, the output vocabulary is restricted to the valid codes observed in the supervised training set. This prevents malformed and out-of-set generations.

All protocols are evaluated on the complete held-out test set. Invalid generations are retained as incorrect predictions and are not excluded from metric calculation.

### Running evaluation

Before evaluation, update `MODEL_PATH`, `ADAPTER_PATH`, `TRAIN_FILE`, `TEST_FILE`, and `OUTPUT_DIR` in the evaluation script.

Run the topography evaluation from a terminal using:

```bash
cd /workspace/Reviewer_comment

python -u \
  /workspace/Reviewer_comment/llama3_70b_topography_final_evaluation.py
```

From a Jupyter notebook, use:

```python
!python -u /workspace/Reviewer_comment/llama3_70b_topography_final_evaluation.py
```

The supplied topography evaluation configuration expects eight visible GPUs and distributes the quantized 70B model across them. Adjust `MAX_MEMORY` and the GPU-count check if a different hardware configuration is used.

### Evaluation outputs

The evaluation script produces:

* Report-level gold labels and predictions for SRG, RFE, and PCD
* Accuracy
* Weighted precision, recall, and F1-score
* Macro precision, recall, and F1-score
* Micro precision, recall, and F1-score
* Invalid-generation count and rate
* Separate results for all reports, classes observed during training, and unseen classes
* A JSON file recording the principal evaluation settings

Malformed outputs are represented internally by a non-clinical invalid label. This label is used to count malformed generations as errors but is excluded from the set of genuine classes used to calculate macro-F1.

### Important notes

* Do not use a morphology adapter for topography evaluation or a topography adapter for morphology evaluation.
* PCD candidates are obtained only from the corresponding supervised training split.
* Ground-truth classes absent from training remain in the held-out test set.
* Use a new output directory when changing the model, adapter, decoding settings, or parsing rules to avoid mixing predictions from different runs.
* Patient reports, dataset files, trained adapters, model checkpoints, and report-level predictions should not be committed to the public repository.
