# CCS5600 Group Assignment 2: Naive Bayesian Anti-Spam Filtering

This repository contains a reproducible implementation of the Ling-Spam Naive Bayes anti-spam filtering assignment.

## Project Structure

```text
data/
  lingspam_public/                  # Ling-Spam corpus with bare, stop, lemm, lemm_stop
src/
  task1_preprocess_all_versions.py  # Data parsing, tokenization, vocabulary summaries
  task2_naive_bayes.py              # Tom M. Mitchell-style Naive Bayes classifier
  task3_metrics.py                  # Cost-sensitive metrics: WAcc, precision, recall, TCR
  task4_training_size_experiment.py # Training-size learning-curve experiment
  task4_generate_final_outputs.py   # Final Python-generated figures and comparison table
results/
  task1_results_all_versions/
  task2_results_mitchell/
  task3_results/
  task4_training_size_results/
  task4_python_outputs/
figures/
  weighted_accuracy_vs_vocab.png
  tcr_vs_vocab.png
  preprocessing_comparison_table.png
  training_size_vs_tcr.png
report/
  Group_Assignment_2_Report_Draft.pdf
  Group_Assignment_2_Report_Draft.docx
docs/
  assignment_brief.pdf
  reference_paper.pdf
```

## Dataset

The Ling-Spam corpus contains 2,893 emails:

- 481 spam emails
- 2,412 ham emails
- 10 predefined folds
- Four preprocessing versions: `bare`, `stop`, `lemm`, and `lemm_stop`

The dataset is acknowledged according to the original Ling-Spam README.

## Environment Setup

Use Python 3.10+.

```powershell
python -m pip install -r requirements.txt
```

Most experiment scripts use only the Python standard library. `matplotlib` is needed for final figure generation.

## Reproducing The Experiments

Run the scripts from the repository root in this order:

```powershell
python src/task1_preprocess_all_versions.py
python src/task2_naive_bayes.py
python src/task3_metrics.py
python src/task4_training_size_experiment.py
python src/task4_generate_final_outputs.py
```

The scripts use repository-relative paths. The expected dataset location is:

```text
data/lingspam_public/
```

That folder must directly contain:

```text
bare/
stop/
lemm/
lemm_stop/
readme.txt
```

## Main Outputs

The final report figures are:

- `figures/weighted_accuracy_vs_vocab.png`
- `figures/tcr_vs_vocab.png`
- `figures/preprocessing_comparison_table.png`
- `figures/training_size_vs_tcr.png`

The final report draft is:

- `report/Group_Assignment_2_Report_Draft.pdf`
- `report/Group_Assignment_2_Report_Draft.docx`

Before final submission, replace title-page placeholders for group number, member names, IDs, lecturer, and submission date.

## Implementation Notes

- Task 1 parses emails and preserves the ten-fold structure.
- Task 2 implements Naive Bayes text classification from scratch using the Tom M. Mitchell formulation.
- Laplace smoothing is used for word probabilities.
- Log probabilities are used to avoid numerical underflow.
- Task 3 applies the assignment's cost-sensitive rule:

```text
P(Spam | Email) / P(Ham | Email) > lambda
```

Using log scores, this becomes:

```text
spam_log_score - ham_log_score > log(lambda)
```

- Task 4 evaluates vocabulary size, preprocessing version, and training-corpus size.
