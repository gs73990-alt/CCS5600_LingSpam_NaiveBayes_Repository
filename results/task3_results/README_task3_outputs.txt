Task 3 Metrics Implementation

This script reads Task 2 prediction files created by the Mitchell Naive Bayes script.
It applies the cost-sensitive decision rule required in the assignment.

Decision rule:
Classify as spam only if P(Spam | Email) / P(Ham | Email) > lambda.

Because Task 2 stored log scores, the script uses:
spam_log_score - ham_log_score > log(lambda).

Lambda values used:
1
9
99

Metrics calculated:
Accuracy
Spam Precision
Spam Recall
Weighted Accuracy
Baseline Weighted Accuracy
Total Cost Ratio
TP, TN, FP, FN

Main file to open first:
overall_task3_metrics.csv
