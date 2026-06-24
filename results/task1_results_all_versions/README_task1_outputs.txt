Task 1 outputs for all Ling-Spam corpus versions

Processed versions:
bare: No lemmatization, stop-words included
stop: No lemmatization, stop-words removed
lemm: Lemmatized, stop-words included
lemm_stop: Lemmatized, stop-words removed

Important files:
combined_dataset_summary.csv: Overall email counts for all four versions.
combined_cross_validation_summary.csv: Training and testing fold summaries for all versions and vocabulary sizes.
Each version folder contains email_metadata.csv, fold_summary.csv, cross_validation_summary.csv, task1_report_summary.txt, vocabulary files, train vectors, and test vectors.

Vector files:
Binary vector files were created for N=100 words.
A value of 1 means the vocabulary word appears in the email.
A value of 0 means the vocabulary word does not appear in the email.

Label encoding:
spam = 1
ham = 0
