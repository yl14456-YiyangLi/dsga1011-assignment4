# DS-GA 1011 Assignment 4 - Yiyang Li (yl14456)

Fall 2026. BERT sentiment robustness, T5 text-to-SQL fine-tuning, and T5 training from random initialization.

## Report and checkpoints

- [Written report](report/A4_Yiyang_Li_Report.pdf)
- [Editable report text](report/A4_Yiyang_Li_Report.md)
- [Fine-tuned T5 checkpoint (Q7)](https://drive.google.com/drive/folders/1cXNUmQSnFv2-i9Mjb8IhlLEBo_Ca8Qq1?usp=sharing)
- [T5 from scratch checkpoint (extra credit)](https://drive.google.com/drive/folders/12agYTVwNu_qbwhKeci8FEhSiHQ-GRs_4?usp=sharing)

Download **all five files in each checkpoint folder** into its own local directory: `config.json`, `generation_config.json`, `model.safetensors`, `tokenizer_config.json`, `tokenizer.json`. Weights are on Drive, not in this repository.

## Results

| BERT training set | Original accuracy | Typo accuracy (p=0.50) |
|---|---:|---:|
| Original 25,000 | 92.896% | 87.808% |
| Original + 5,000 transformed | 92.912% | 89.820% |

| Model | Dev Record F1 | Dev Record EM | Dev SQL EM | Test Record F1 | Test Record EM |
|---|---:|---:|---:|---:|---:|
| T5 fine-tuned, epoch 20 | 86.8417% | 84.7639% | 69.7425% | 81.65% | 79.17% |
| T5 random initialization, epoch 50 | 63.8617% | 60.5150% | 45.2790% | 54.31% | 51.16% |

Test results are from Gradescope submission 449594989. Gradescope displays test SQL Query EM as 0.00 for both models; this value is reproduced in the report without inferring a cause. Hidden gold SQL was not available for independent checking. The page reports 25/25 Q7 performance points and that the extra-credit F1 >= 50 threshold is met; final grades are assigned by course staff.

## Environment

Recorded training environment: Python 3.13.16, PyTorch 2.14.1+cu130, Transformers 5.22.1, Datasets 4.8.5, Evaluate 0.4.6, NVIDIA A100-SXM4-40GB; FP32. Install a suitable CUDA build of PyTorch for your machine, then the dependencies in `requirements.txt`. Exact GPU/runtime differences can affect training and SQL timeout behavior.

## Part 1

Run from `part1/`. The dataset is downloaded from `stanfordnlp/imdb`.

```bash
cd part1
# Optional initial smoke run; the full run below starts from pretrained BERT again.
python main.py --train --eval --debug_train
# Q1: fresh BERT on 25k examples; saves out/ and out_original.txt.
python main.py --train --eval
# Q2: deterministic internal adjacent-letter swaps, p=0.50.
python main.py --eval_transformed --model_dir out
# Q3: fresh BERT on 25k + 5k, then both test sets.
python main.py --train_augmented --eval --eval_transformed
```

All runs use `bert-base-cased`, batch 8, 3 epochs, learning rate 5e-5 and a linear schedule without warmup. The transformation in `utils.py` uses a per-text SHA-256 seed, never the label. `results/part1/augmentation_indices.json` contains the 5,000 selected indices. The exploratory p=0.30 result is retained for transparency; the submitted transformation is p=0.50.

## Part 2 setup

Use the original starter ZIP provided with the assignment (or downloaded from the [course starter repository](https://github.com/gregdurrett/nyu-dsga1011-a4)). Data and the SQLite database are not redistributed here.

```bash
cd part2
python prepare_data.py /path/to/starter_original.zip
python prepare_tokenizer.py
python compute_statistics.py
```

`prepare_tokenizer.py` adds `<`, `<=`, and `<>` as ordinary tokens and saves `tokenizer_sql/`. Training files are read without lowercasing, schema prompts, filtering, or truncation. Length statistics in the report exclude EOS, while training sequences include EOS. Dynamic right padding uses token 0; decoder inputs are shifted right, starting at 0; pad targets are excluded from the loss.

Run the following from **part2/** so that the evaluator can find `data/` and `records/`. Output directories must be new.

### Fine-tuning

```bash
python finetuned/train_t5.py --finetune --output_dir ../runs/t5_ft \
  --learning_rate 0.0003 --batch_size 8 --test_batch_size 16 \
  --max_n_epochs 20 --patience_epochs 5 --scheduler_type linear \
  --num_warmup_epochs 1 --weight_decay 0.01 --max_new_tokens 512 --seed 42
```

### Extra credit: fresh random weights

```bash
python scratch/train_t5.py --output_dir ../runs/t5_scratch \
  --learning_rate 0.0003 --dropout_rate 0 --weight_decay 0 \
  --batch_size 16 --test_batch_size 16 --max_n_epochs 60 \
  --patience_epochs 10 --scheduler_type linear --num_warmup_epochs 1 \
  --max_new_tokens 512 --seed 42
```

The scratch branch constructs `T5ForConditionalGeneration(config)` and does **not** load pretrained weights. Architecture dimensions match t5-small; vocabulary size is set to the 32,103-token tokenizer before construction. The original Colab run read the architecture JSON from the fine-tuned checkpoint; the portable script defaults to the equivalent t5-small configuration. To reuse that exact JSON, set `T5_CONFIG_PATH` to the downloaded fine-tuned checkpoint folder. This only reads configuration, not weights.

Both trainers save the best dev Record F1 checkpoint. The scratch trainer permits early stopping from epoch 20 onward. It ran 60 epochs, selected epoch 50, and printed early stopping when patience was reached at epoch 60. Only model/tokenizer checkpoints are saved, so exact optimizer/scheduler-state resume is not implemented.

### Final test generation

Use the released checkpoints to reproduce the submitted models without training again:

```bash
python generate_test.py --checkpoint /path/to/finetuned_best --output_dir ../runs/ft_test
python generate_test.py --checkpoint /path/to/scratch_best --output_dir ../runs/ec_test --extra_credit
```

Greedy decoding, at most 512 new tokens, batch 16, no reranking or post-hoc SQL repair. Query execution is read-only with four workers and a 10-second per-query budget. The pickle is a tuple `(records, error_messages)` in test-file order. Avoid repeatedly probing the hidden test set.

## Audit and provenance

- `results/`: original metrics, configs and histories; `run_config.json` and `command.json` intentionally retain historical Colab paths as evidence.
- `analysis/`: all 466 dev examples for each model with SQL, errors and explicit classification; counts are in `error_summary.json`.
- `predictions/`: the eight submitted files, unchanged from the reviewed run exports.
- `scripts/validate_outputs.py`: checks prediction counts, Part 1 accuracy, and audit counts without loading model weights.
- `results/scratch_initial_failed/`: unsuccessful first scratch configuration; it was not submitted for hidden-test evaluation.

The packaged training loops and evaluator are the run-exported implementations. Portability edits replace hard-coded data/tokenizer/config paths with repository-relative defaults or environment overrides and remove an unused wandb import. `prepare_data.py`, `prepare_tokenizer.py`, and the consolidated test-generation entry point provide standalone setup. No submitted predictions were regenerated or altered while packaging. CPU validation covers syntax and saved outputs; full GPU retraining was not repeated during packaging.

The before/after scratch experiments change several settings together; they are not a dropout-only ablation. Small-sample diagnostics are training fit checks, never dev/test scores. See the report for limitations and error examples.

## Submission

Coding: upload the **eight files inside `predictions/` directly**, preserving names. Written: submit `report/A4_Yiyang_Li_Report.pdf` and assign its pages to the corresponding questions if Gradescope asks. The GitHub repository and both model links appear in the PDF.

## Attribution

Built on [gregdurrett/nyu-dsga1011-a4](https://github.com/gregdurrett/nyu-dsga1011-a4), the Fall 2026 assignment starter. Uses Hugging Face `bert-base-cased`, `google-t5/t5-small`, and `stanfordnlp/imdb`. The supplied evaluation metric definitions are retained.
