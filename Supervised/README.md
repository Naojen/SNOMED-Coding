## Supervised Fine-Tuning and Evaluation

The supervised experiments in the study evaluated 17 large language models. This repository provides representative training and evaluation implementations for two of these models:

* **LLaMA-3-70B-Instruct**, representing a general-purpose LLM.
* **MedGemma-27B-text-it**, representing a medical-domain LLM.

For each model, separate scripts are provided for SNOMED morphology and topography coding.

| Model                | Task       | Training script                         | Evaluation script                             |
| -------------------- | ---------- | --------------------------------------- | --------------------------------------------- |
| LLaMA-3-70B-Instruct | Morphology | `llama3_70b_morphology_qlora_full.py`   | `llama3_70b_morphology_final_evaluation.py`   |
| LLaMA-3-70B-Instruct | Topography | `llama3_70b_topography_qlora_full.py`   | `llama3_70b_topography_final_evaluation.py`   |
| MedGemma-27B-text-it | Morphology | `medgemma_27b_morphology_qlora_full.py` | `medgemma_27b_morphology_final_evaluation.py` |
| MedGemma-27B-text-it | Topography | `medgemma_27b_topography_qlora_full.py` | `medgemma_27b_topography_final_evaluation.py` |

These scripts illustrate the complete supervised fine-tuning and three-protocol evaluation workflow used in the study. The remaining models were trained and evaluated using the same task formulation and experimental protocol, with model-specific adjustments to loading and distributed execution.

### Training configuration

All supervised models were fine-tuned using 4-bit Quantized Low-Rank Adaptation (QLoRA) with the following configuration:

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
* Maximum sequence length: 512
* Maximum training duration: 15 epochs
* Model-selection criterion: lowest validation loss

Gradient checkpointing was enabled to reduce GPU-memory consumption. Gradient accumulation was adjusted according to the number of active GPUs so that the global effective batch size remained fixed at 32.

The supplied four-GPU training scripts use a per-device batch size of 1 and eight gradient-accumulation steps:

```text
1 sample per GPU × 4 GPUs × 8 accumulation steps = 32
```

The LLaMA-3-70B scripts use Fully Sharded Data Parallelism (FSDP). The MedGemma-27B scripts use Distributed Data Parallelism (DDP), with one quantized model copy loaded on each GPU.

### Data format

Training, validation, and test files use JSON Lines format. Each record must contain `instruction`, `input`, and `output` fields:

```json
{
  "instruction": "Assign the appropriate SNOMED morphology code.",
  "input": "Pathology report text",
  "output": "81403"
}
```

For topography coding, the instruction and output must correspond to the topography task. Morphology and topography models are trained separately using their respective dataset partitions:

```text
morphology_train.jsonl
morphology_validation.jsonl
morphology_test.jsonl

topography_train.jsonl
topography_validation.jsonl
topography_test.jsonl
```

The formatting used during training is:

```text
### Instruction:
{instruction}

### Pathology report:
{input}

### Response:
MORPHOLOGY_CODE: {output}
```

For topography training, the response label is:

```text
TOPOGRAPHY_CODE: {output}
```

### Running QLoRA training

Before running a script, update `MODEL_PATH`, `TRAIN_FILE`, `VALIDATION_FILE`, and `OUTPUT_DIR` according to the selected model and task.

A generic Accelerate launch command is:

```bash
cd /workspace/Reviewer_comment

ACCELERATE_CONFIG=/path/to/accelerate_config.yaml
TRAINING_SCRIPT=/path/to/model_task_qlora_full.py

accelerate launch \
  --config_file "$ACCELERATE_CONFIG" \
  "$TRAINING_SCRIPT"
```

For example, select one of the following training scripts:

```text
llama3_70b_morphology_qlora_full.py
llama3_70b_topography_qlora_full.py
medgemma_27b_morphology_qlora_full.py
medgemma_27b_topography_qlora_full.py
```

Use an FSDP Accelerate configuration for the supplied LLaMA-3-70B scripts and a DDP Accelerate configuration for the supplied MedGemma-27B scripts.

The checkpoint having the lowest validation loss is restored after training and saved in the output location defined by the corresponding script.

### Evaluation protocols

Each supervised model is evaluated under three inference protocols:

1. **Strict Raw Generation (SRG):** Generation is unconstrained, and the output is considered valid/invalid? Wait, user wants readable. Need finish.

2. **Strict Raw Generation (SRG):** Generation is unconstrained, and the output is considered valid only when the generated suffix contains exactly one standalone five-digit code. Outputs containing additional labels, explanations, punctuation, or malformed continuations are treated as incorrect.

3. **Relaxed First-Code Extraction (RFE):** Generation remains unconstrained, but the first standalone five-digit code is extracted from the generated response. A response containing no valid five-digit code is treated as incorrect.

4. **Prefix-Constrained Decoding (PCD):** During decoding, the permitted output vocabulary is restricted to valid codes observed in the corresponding supervised training split. This prevents malformed and out-of-set code generation.

All three protocols are evaluated on the complete held-out test set. Invalid generations are retained as incorrect predictions and are not excluded from metric calculation. Ground-truth classes absent from the training split also remain in the test set.

### Running evaluation

Before evaluation, update `MODEL_PATH`, `ADAPTER_PATH`, `TRAIN_FILE`, `TEST_FILE`, and `OUTPUT_DIR` in the selected evaluation script.

A generic evaluation command is:

```bash
cd /workspace/Reviewer_comment

EVALUATION_SCRIPT=/path/to/model_task_final_evaluation.py

python -u "$EVALUATION_SCRIPT"
```

From a Jupyter notebook, set the desired script path and run:

```python
EVALUATION_SCRIPT = "/path/to/model_task_final_evaluation.py"
!python -u "$EVALUATION_SCRIPT"
```

Select one of the following evaluation scripts:

```text
llama3_70b_morphology_final_evaluation.py
llama3_70b_topography_final_evaluation.py
medgemma_27b_morphology_final_evaluation.py
medgemma_27b_topography_final_evaluation.py
```

The supplied LLaMA-3-70B evaluation scripts distribute the quantized model across eight visible GPUs. The supplied MedGemma-27B evaluation scripts load the quantized model on a single GPU. The device mapping and memory limits can be adjusted for other hardware configurations.

### Evaluation outputs

Each evaluation script produces:

* Report-level gold labels and predictions for SRG, RFE, and PCD
* Accuracy
* Weighted precision, recall, and F1-score
* Macro precision, recall, and F1-score
* Micro precision, recall, and F1-score
* Invalid-generation count and rate
* Separate results for all reports, classes observed during training, and classes not observed during training
* A JSON file recording the principal evaluation settings

Malformed outputs are represented internally by a non-clinical invalid label. This label ensures that malformed generations are counted as errors but is excluded from the set of genuine SNOMED classes used to calculate macro-F1.

### Important notes

* Do not use a morphology adapter for topography evaluation or a topography adapter for morphology evaluation.
* PCD candidates are obtained only from the corresponding supervised training split.
* Ground-truth classes absent from training remain in the held-out test set.
* Use a new output directory after changing the model, adapter, decoding settings, or parsing rules to prevent predictions from different runs from being combined.
* Model and task paths must be checked before every training or evaluation run.
* Patient reports, dataset files, trained adapters, model checkpoints, and report-level predictions must not be committed to the public repository.
