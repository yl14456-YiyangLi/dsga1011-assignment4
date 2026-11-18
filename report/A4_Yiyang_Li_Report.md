Assignment 4: Fine-tuning Language Models

DS-GA 1011 Natural Language Processing | Fall 2026<br/>Yiyang Li (yl14456)

## Part 1. Robust Sentiment Classification

### Q1. Fine-tuning BERT

I fine-tuned **bert-base-cased** for binary sentiment classification on the 25,000 IMDB training reviews. All model parameters were updated. Reviews were tokenized with the cased BERT tokenizer, truncated to 512 tokens, and padded to a fixed length of 512. The full-data model obtained **92.896% accuracy** on the 25,000 original test reviews, exceeding the 91% requirement.

Training used AdamW, learning rate 5e-5, batch size 8, three epochs, a linear learning-rate schedule without warmup, and FP32 arithmetic. AdamW used its default weight decay 0.01, betas (0.9, 0.999), and epsilon 1e-8. Python and PyTorch seeds were set to 0, with deterministic cuDNN settings. The final three-epoch checkpoint was used for the reported evaluations; the transformed test set was not used as training data.

### Q2a. A deterministic typographical transformation

The transformation models internal adjacent-letter typing mistakes. It scans alphabetic words using the regular expression **\b[A-Za-z]+\b**. Words shorter than five characters are unchanged, and the negation-related words never, neither, nor, not, no, and without are protected. For each remaining word, a Bernoulli draw with probability **p = 0.50** determines whether to attempt a typo. One eligible pair of adjacent, unequal letters is selected uniformly from the interior and swapped. The first and last characters are preserved. Words with no eligible pair remain unchanged.

The random generator is initialized separately for each review using the first eight bytes of its SHA-256 digest, interpreted as a big-endian integer. Thus, rerunning the function on the same review gives the same output and requires no global RNG state. Labels never enter the transformation. All other dataset fields, punctuation, and spacing are retained. For example, swapping the internal “r” and “e” in “great” yields “gerat”; that is an illustration of an allowed edit, not a claim about a particular saved review.

The intended sentiment is usually recoverable from the surrounding review, while BERT sees less familiar subword sequences. Protecting negation reduces the risk of reversing polarity. Nevertheless, individual swaps can create real words or reduce readability, so label preservation is a practical assumption rather than a guarantee. A 50% selection probability is a controlled robustness stress test, not an estimate of naturally occurring typo frequency.

### Q2b. Robustness before augmentation

An exploratory p = 0.30 transformation gave 90.416% accuracy, a decrease of 2.480 percentage points. The final p = 0.50 transformation gave **87.808%**, a decrease of **5.088 percentage points** from 92.896%, satisfying the required drop of more than four points. Both settings are retained in the experiment records; the p = 0.50 outputs were submitted.


---


## Part 1. Data Augmentation

### Q3. Training on original and transformed examples

I sampled exactly 5,000 distinct indices from the 25,000 original training examples using random.Random(42).sample. I applied the final p = 0.50 transformation to those reviews, retained their labels, and concatenated the transformed examples with all 25,000 originals. The resulting training set had 30,000 examples. A new classifier was initialized from pretrained bert-base-cased and trained with the same learning rate, batch size, and three-epoch schedule as Q1. It was not initialized from the Q1 fine-tuned checkpoint.

| Training data | Original test (%) | Transformed test (%) | Drop (points) |
| --- | --- | --- | --- |
| Original 25,000 | 92.896 | 87.808 | 5.088 |
| Original + 5,000 transformed | 92.912 | 89.820 | 3.092 |

Table 0. BERT accuracies on 25,000 test reviews. The transformed column uses p = 0.50 in both rows.

Augmentation improved transformed-test accuracy by **2.012 points** (503 additional correct predictions), while original-test accuracy changed by only **0.016 points** (four additional correct predictions). The original-to-transformed gap shrank from 5.088 to 3.092 points. This is consistent with learning greater tolerance to the specific typo pattern without materially reducing clean-input accuracy.

The remaining gap shows that augmentation did not eliminate sensitivity to spelling changes. The transformed test distribution uses the same transformation family as the added training data; these results do not establish robustness to unrelated corruptions, paraphrases, or adversarial attacks. Only one augmentation sample and training seed were evaluated, so I do not interpret the four-example clean-set increase as statistically established improvement.

## Part 2. Text-to-SQL with T5

### Q4. Data before and after preprocessing

The flight-domain dataset contains 4,225 training examples, 466 development examples, and 432 unlabeled test inputs. Each labeled example pairs a natural-language question with a SQL query. The following statistics use the T5 tokenizer, without padding or special tokens. “Vocabulary” means the number of distinct token IDs observed in that column and split, not the tokenizer’s full vocabulary size.

| Split | Examples | Mean NL
length | Mean SQL
length | NL vocab. | SQL vocab. |
| --- | --- | --- | --- | --- | --- |
| Train | 4,225 | 17.097 | 216.373 | 791 | 555 |
| Dev | 466 | 17.071 | 210.054 | 465 | 395 |

Table 1. Token statistics before adding SQL operators.

| Split | Examples | Mean NL
length | Mean SQL
length | NL vocab. | SQL vocab. |
| --- | --- | --- | --- | --- | --- |
| Train | 4,225 | 17.097 | 216.207 | 791 | 556 |
| Dev | 466 | 17.071 | 209.891 | 465 | 396 |

Table 2. Token statistics after adding SQL operators.

I added &lt;, &lt;=, and &lt;&gt; as ordinary, non-special AddedTokens with normalization disabled. This increased the tokenizer size from 32,100 to 32,103. Before the change, SQL contained 510 unknown tokens in 393 training examples and 55 in 45 dev examples; afterward, both splits had zero unknown SQL tokens. Natural-language inputs already had zero unknown tokens. The distinct observed SQL vocabulary increases by only one because observed IDs and the full tokenizer inventory are different quantities. No examples were removed.


---


## Part 2. Model and Training

### Q5. Architecture, preprocessing, and hyperparameters

The model is the pretrained google-t5/t5-small encoder-decoder: six encoder layers, six decoder layers, hidden dimension 512, feed-forward dimension 2,048, and eight attention heads. The shared token embeddings and output vocabulary were resized to 32,103. I updated all parameters. Inputs and targets were stripped of leading/trailing whitespace, but not lowercased or rewritten; no schema prompt or task prefix was added.

Training tokenization appends EOS (ID 1). Batches are padded dynamically on the right with ID 0. An encoder attention mask excludes input padding. Decoder inputs are the targets shifted right, beginning with decoder-start ID 0; cross-entropy is computed only for non-padding target positions. Maximum SQL lengths before EOS are 510 tokens in train and 502 in dev, so no SQL truncation was needed. Encoder lengths are at most 59 and 43 tokens before EOS, respectively.

| Setting | Fine-tuned T5 |
| --- | --- |
| Initialization / trainable parameters | Pretrained t5-small; all parameters trainable |
| Training / evaluation batch size | 8 / 16 |
| Optimizer | AdamW; betas (0.9, 0.999), epsilon 1e-8 |
| Learning rate / weight decay | 3e-4 / 0.01; bias and layer-norm weights excluded from decay |
| Schedule / warmup | Linear decay over 20 epochs / 1 epoch |
| Dropout / gradient clipping | 0.1 / global gradient norm 1.0 |
| Epoch budget / early stopping | 20 / patience 5 on dev Record F1 |
| Random seed / precision | 42 / FP32 |
| Decoding | Greedy; one beam; no sampling; maximum 512 new tokens |
| Model selection | Highest dev Record F1; epoch 20 selected |

Table 3. Reproducible fine-tuning configuration.

Generation begins with the decoder-start token and does not use a reference SQL prefix. There is no reranking, constrained decoding, or post-hoc SQL repair. Predicted queries are executed against the course SQLite database in read-only mode, using four workers and a 10-second per-query budget. Each test prediction is saved in input order as SQL and as a pair of record/error lists in the required pickle format.

The recorded runtime was Python 3.13.16, PyTorch 2.14.1+cu130, Transformers 5.22.1, Datasets 4.8.5, and Evaluate 0.4.6, on an NVIDIA A100-SXM4-40GB. The repository contains setup, training, generation, and artifact-validation commands. Hardware and query timing can affect timeout outcomes, even with a fixed training seed.

### Q6. Experiments and checkpoint selection

I trained one full fine-tuning configuration. Before training, the pipeline passed decoder-shift alignment, GPU forward/backward, checkpoint save/load consistency, and evaluator self-checks. The operator change was verified by token round trips and unknown-token counts. These checks establish correctness properties; they are not a controlled downstream accuracy ablation.

The within-run dev Record F1 rose from 37.2225% at epoch 1 to 75.6386% at epoch 5, 80.4810% at epoch 10, and 86.8417% at epoch 20. These are checkpoints from one optimization run, not separate hyperparameter experiments. The highest dev-F1 checkpoint was selected before generating the final test files.


---


## Part 2. Evaluation and Error Analysis

| Split / model | Record F1 (%) | Record EM (%) | SQL EM (%) | SQL errors |
| --- | --- | --- | --- | --- |
| Dev / fine-tuned, epoch 20 | 86.8417 | 84.7639 | 69.7425 | 25 / 466 |
| Test / same checkpoint | 81.65 | 79.17 | 0.00* | 39 / 432 |

Table 4. Development results from the saved run and test results reported by Gradescope.

*Gradescope displays test SQL Query EM as 0.00 for both submitted T5 models. I reproduce that displayed value without assigning a cause: hidden gold SQL was not available for independent verification. Development SQL EM uses the provided evaluator’s exact string comparison. SQL-string equality and equality of returned records are distinct metrics.

I audited all 466 development predictions. Categories below are mutually exclusive: an execution error takes precedence; otherwise, a record-set mismatch is classified as executable but incorrect. SQL examples are abbreviated, and example numbers are one-based positions in dev.nl. All full queries and error messages are included in analysis/finetuned_dev_errors.json.

| Error type | Count / all dev | Representative example and explanation |
| --- | --- | --- |
| Malformed SQL | 12 / 466; (2.58%) | Example 29: the question asks for United flight 415 on Thursday night. The prediction contains an invalid time condition and extra closing parentheses: departure_time < 415 ) ) ). SQLite reports a syntax error near ")". |
| Schema / identifier | 8 / 466; (1.72%) | Example 79: “most expensive way” to fly to Washington. An outer WHERE clause references airport_service_2.airport_code, although that alias is declared only in an inner query. SQLite reports no such column. |
| Executable, wrong records | 52 / 466; (11.16%) | Example 19: lowest round-trip fare from Dallas to any city. The prediction adds a destination-city restriction to DALLAS, whereas the gold query restricts the origin. It returns zero records instead of four. |
| Execution timeout | 5 / 466; (1.07%) | Example 322: round-trip fares Denver–Pittsburgh. The prediction introduces extra flight_fare_2 and fare_2 tables without corresponding joins. The saved execution is interrupted at the time budget. |

Table 5. Fine-tuned model: four error classes, their frequencies, and concrete examples.

The remaining 389/466 predictions execute successfully and return exactly the gold record set. Six execution failures also receive a record match under the supplied metric because both stored records and gold records are empty; therefore, 395/466 gives the reported Record EM, whereas only 389 are successful executions with matching records. This explains why the 77 error-category entries do not equal the 71 record mismatches.

Semantic errors include entity-role confusion, wrong time constraints, and incorrect aggregation. For example, example 21 replaces a maximum-stops requirement with stops = 1; example 23 predicts a contradictory arrival-time interval. Separately, 66 queries execute without error and return the correct records despite differing from the reference SQL. Thus, exact SQL matching can reject useful alternative formulations, while record matching on one database can also conceal errors, especially for empty results.

### Q7. Hidden-test submission

The selected checkpoint generated 432 nonempty test queries. Gradescope reports **Record F1 = 81.65%** and **25.00/25 performance points**, exceeding the 65% F1 threshold. The test execution-error rate is 39/432 = 9.03%. These automated numbers are informational; final grading is performed by the course staff. The submitted files are t5_ft_experiment_test.sql and t5_ft_experiment_test.pkl.


---


## Extra Credit. T5 from Random Initialization

For the extra-credit experiment, I used the same t5-small architecture and the same expanded tokenizer as in Part 2, but initialized **all model weights randomly** through T5ForConditionalGeneration(config). Reading an architecture configuration does not load a checkpoint’s weights. The tokenizer was reused; pretrained neural parameters were not. Training used only the original 4,225 training pairs.

### EC data statistics and preprocessing

The train/dev splits and tokenization are identical to those used in fine-tuning. Tables EC1 and EC2 are repeated to make the extra-credit experiment self-contained. Lengths exclude EOS and padding; observed vocabularies are counted separately by split and column.

| Split | Examples | Mean NL
length | Mean SQL
length | NL vocab. | SQL vocab. |
| --- | --- | --- | --- | --- | --- |
| Train | 4,225 | 17.097 | 216.373 | 791 | 555 |
| Dev | 466 | 17.071 | 210.054 | 465 | 395 |

Table EC1. Extra-credit data before SQL-operator preprocessing.

| Split | Examples | Mean NL
length | Mean SQL
length | NL vocab. | SQL vocab. |
| --- | --- | --- | --- | --- | --- |
| Train | 4,225 | 17.097 | 216.207 | 791 | 556 |
| Dev | 466 | 17.071 | 209.891 | 465 | 396 |

Table EC2. Extra-credit data after SQL-operator preprocessing.

The same three ordinary SQL tokens remove all unknown SQL tokens. No lowercasing, schema prompt, filtering, or truncation is applied. Training appends EOS, dynamically pads with ID 0, shifts decoder targets right, and excludes padding from cross-entropy. This keeps the input/output representation comparable across pretrained and randomly initialized models.

| Setting | Final extra-credit configuration |
| --- | --- |
| Architecture / vocabulary | T5-small: 6 encoder + 6 decoder layers; d_model 512; d_ff 2,048; 8 heads; vocabulary 32,103 |
| Initialization | Fresh random weights for all parameters |
| Training / evaluation batch size | 16 / 16 |
| Optimizer / learning rate | AdamW, betas (0.9, 0.999), epsilon 1e-8 / 3e-4 |
| Weight decay / dropout | 0 / 0 |
| Schedule / gradient clipping | Linear; 1 warmup epoch / norm 1.0 |
| Budget / early stopping | 60 epochs; patience 10; stopping allowed from epoch 20 |
| Seed / precision / selection | 42 / FP32 / highest dev Record F1, epoch 50 |
| Decoding / execution | Greedy, 512 new tokens; same SQLite evaluation as Part 2 |

Table EC3. Architecture and hyperparameters for training T5 from scratch.


---


## Extra Credit. Experiments and Results

### An unsuccessful configuration and diagnostic checks

The first full scratch run used learning rate 1e-3, dropout 0.1, weight decay 0.01, batch size 16, two warmup epochs, and a 60-epoch budget. It stopped at epoch 25 after ten epochs without a dev-F1 improvement. Its selected epoch-15 checkpoint had Record F1 11.8026%, SQL EM 0%, and 464/466 execution errors. Generated SQL often repeated table names. The small nonzero record score is misleading here: the development set contains 55 empty gold result sets, and failed queries are stored as empty records.

I first compared cached and uncached generation on a diagnostic example. The generated token IDs were identical. Prefix checks at lengths 8, 16, and 24 had argmax agreement 1.0 and maximum absolute logit differences no larger than approximately 1.5e-5. These checks did not support a cache inconsistency for the tested inputs, but they do not prove every possible implementation path correct.

Next, I fitted eight fixed training examples from fresh random initialization. At learning rate 3e-4 with the initial dropout setting, the model exactly generated only one of eight examples at step 500 (best: two at step 400), despite low teacher-forced loss. Repeating the small fit check with dropout zero reached six of eight at step 200 and eight of eight at step 300, with teacher-forced token accuracy 1.0 and loss 0.000537. A planned learning-rate reduction at step 501 was never reached. These are training-set memorization diagnostics, not development or test performance.

I then ran the final full-data configuration in Table EC3. Learning rate, dropout, weight decay, and warmup changed together relative to the failed full run. Therefore, the final improvement cannot be attributed to dropout alone. The small-fit result justified proceeding with full-data training, but did not establish generalization.

| Run / checkpoint | Dev F1 (%) | Dev EM (%) | Dev SQL EM (%) |
| --- | --- | --- | --- |
| Initial scratch, best epoch 15 | 11.8026 | 11.8026 | 0.0000 |
| Final scratch, epoch 20 | 50.0935 | 46.1373 | 31.3305 |
| Final scratch, epoch 40 | 60.8364 | 57.2961 | 43.3476 |
| Final scratch, best epoch 50 | 63.8617 | 60.5150 | 45.2790 |
| Final scratch, epoch 60 | 63.6465 | 60.3004 | 46.1373 |

Table EC4a. Actual scratch experiments and within-run checkpoints. Rows from the final run are not independent hyperparameter trials.

| Split / selected checkpoint | Record F1 (%) | Record EM (%) | SQL EM (%) | SQL errors |
| --- | --- | --- | --- | --- |
| Dev / final scratch epoch 50 | 63.8617 | 60.5150 | 45.2790 | 19 / 466 |
| Test / same checkpoint | 54.31 | 51.16 | 0.00* | 34 / 432 |

Table EC4b. Final extra-credit development and hidden-test results. *Same Gradescope reporting caveat as Table 4.

The run completed 60 epochs. The selected checkpoint is epoch 50, not epoch 60: no later epoch improved dev Record F1, and the patience condition was reached at epoch 60. Test generation produced 432 nonempty queries with 34 execution errors (7.87%). Gradescope reports **Record F1 = 54.31%** and explicitly marks the **F1 &gt;= 50 extra-credit threshold as met**. The files are t5_ft_experiment_ec_test.sql and t5_ft_experiment_ec_test.pkl.


---


## Extra Credit. Error Analysis and Comparison

I applied the same error-category precedence to all 466 development predictions from the selected scratch checkpoint. Full examples and saved execution outcomes appear in analysis/scratch_dev_errors.json. Frequencies below use all development queries as the denominator, not only the incorrect subset.

| Error type | Count / all dev | Representative example and explanation |
| --- | --- | --- |
| Malformed SQL | 14 / 466; (3.00%) | Example 26: flights between 10 AM and 3 PM from Pittsburgh to Fort Worth. The generated city literal begins 'FORT WHERE ... and is not closed correctly. The gold literal is 'FORT WORTH'. |
| Schema / identifier | 4 / 466; (0.86%) | Example 98: Pittsburgh airport-to-downtown transport and cost. The FROM clause declares airport_service_2, but a WHERE condition references airport_service_1. The undeclared alias produces a no-such-column error. |
| Executable, wrong records | 168 / 466; (36.05%) | Example 6: “i'm starting from denver.” The prediction constrains both origin and destination to DENVER, while the reference constrains only the origin. It removes valid departing flights. |
| Execution timeout | 1 / 466; (0.21%) | Example 353: Indianapolis–Orlando flights on December 27 with fares. The predicted SQL equals the gold SQL, but its saved execution was interrupted. This is a runtime-budget failure, not evidence of incorrect SQL semantics. |

Table EC5. Scratch model: four error classes, frequencies, and development examples.

There are 279/466 successful executions with exactly matching records. Three execution failures also match empty gold records under the supplied metric, producing Record EM = 282/466 = 60.5150%. Thus, the 187 examples in the four error categories correspond to 184 record mismatches. Another 69 successful queries return the correct records despite differing from the gold SQL string.

The largest difference from fine-tuning is the number of executable queries with wrong records: **168 versus 52**. Many predictions have plausible SQL structure but misinterpret the question’s entities or constraints. Example 11 reverses the origin and destination for an Atlanta-to-Baltimore request. This distinction matters because a low SQL execution-error rate alone does not demonstrate good semantic parsing.

The final scratch model has fewer test execution errors than the fine-tuned model (34 versus 39), yet its test Record F1 is much lower (54.31% versus 81.65%). Pretraining is associated with a 27.34-point test-F1 advantage in these experiments. The dev-F1 difference is approximately 22.98 points. Because the two runs also use different optimization settings and training budgets, this is an empirical comparison of the submitted systems, not an isolated causal estimate of pretraining.

The scratch model’s train loss at the selected epoch is approximately 0.00135, compared with dev loss 0.03661. By epoch 60, train loss falls further while dev Record F1 remains below its epoch-50 peak. This supports selecting by held-out generation quality rather than training loss. It also illustrates why the earlier eight-example fit check was useful for debugging but insufficient for predicting full-task accuracy.

### Limitations and evaluation safeguards

All hyperparameter and checkpoint decisions described here use training diagnostics or development results. Each final system’s reported hidden-test score comes from the submitted prediction files. Record metrics use the supplied evaluator, including its behavior on empty records; execution errors are reported separately to expose that limitation. Query timeouts can vary with hardware and database load. No statistical significance claim is made from the single-seed comparisons.


---


## Code, Checkpoints, and Reproduction

The code repository contains Part 1 training and transformation code, Part 2 fine-tuning and scratch training code, setup and test-generation entry points, original run metrics, development error audits, and all eight submitted prediction files. README.md gives commands and expected outputs. Model weights are provided separately through the following Drive folders.

**GitHub repository**<br/><link href="https://github.com/yl14456-YiyangLi/dsga1011-assignment4" color="#184a75">https://github.com/yl14456-YiyangLi/dsga1011-assignment4</link>

**Fine-tuned T5: best checkpoint (epoch 20)**<br/><link href="https://drive.google.com/drive/folders/1cXNUmQSnFv2-i9Mjb8IhlLEBo_Ca8Qq1?usp=sharing" color="#184a75">https://drive.google.com/drive/folders/1cXNUmQSnFv2-i9Mjb8IhlLEBo_Ca8Qq1?usp=sharing</link>

**Extra credit: scratch T5 best checkpoint (epoch 50)**<br/><link href="https://drive.google.com/drive/folders/12agYTVwNu_qbwhKeci8FEhSiHQ-GRs_4?usp=sharing" color="#184a75">https://drive.google.com/drive/folders/12agYTVwNu_qbwhKeci8FEhSiHQ-GRs_4?usp=sharing</link>

Each model folder contains config.json, generation_config.json, model.safetensors, tokenizer_config.json, and tokenizer.json. Download the complete folder so that the model and tokenizer use the same vocabulary. The scratch checkpoint contains randomly initialized then trained weights, not the small-sample diagnostic checkpoint or the failed first-run checkpoint.

| Material | Location in repository |
| --- | --- |
| Written report | report/A4_Yiyang_Li_Report.pdf |
| Part 1 implementation | part1/main.py; part1/utils.py |
| T5 implementations | part2/finetuned/; part2/scratch/ |
| Preparation and generation | part2/prepare_data.py; prepare_tokenizer.py; generate_test.py |
| Raw experiment records | results/part1/; results/q4/; results/finetuned/; results/scratch/ |
| Development error analysis | analysis/finetuned_dev_errors.json; scratch_dev_errors.json; error_summary.json |
| Submitted predictions | predictions/ (four sentiment text files and two SQL/pickle pairs) |
| Saved-output validation | scripts/validate_outputs.py |

Reproduction index. Historical Colab paths remain in original command/config JSON files as provenance.

### Reproduction procedure

1. Install the recorded dependencies and obtain the course starter data. For Part 1, run the baseline, transformed evaluation, and augmented training commands in README.md. The 5,000 augmentation indices are included for comparison.<br/>2. For Part 2, run prepare_data.py on the original starter ZIP and prepare_tokenizer.py. Execute the listed fine-tuning or scratch training command from part2/.<br/>3. To evaluate the released models directly, download a complete best-checkpoint folder and run generate_test.py with that path. Use --extra_credit for the scratch model.<br/>4. Run scripts/validate_outputs.py to verify saved prediction counts, sentiment accuracies, and error-analysis totals without loading model weights.

Packaging changes only make paths portable, remove an unused import, and add setup/generation helpers. The exported training loops and metric definitions are retained, and the submitted prediction bytes are unchanged. Checkpoints save model/tokenizer state; exact optimizer/scheduler-state resumption is not implemented. Full GPU retraining was not repeated during final packaging.

### References and attribution

Course materials: DS-GA 1011, Assignment 4: Fine-tuning Language Models, Fall 2026; starter repository <link href="https://github.com/gregdurrett/nyu-dsga1011-a4" color="#184a75">gregdurrett/nyu-dsga1011-a4</link>. Models and data: Hugging Face bert-base-cased, google-t5/t5-small, and stanfordnlp/imdb. Official hidden-test numbers are transcribed from Gradescope submission 449594989.
