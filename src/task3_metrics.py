from pathlib import Path
import csv
import math

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK2_RESULTS_DIR = REPO_ROOT / "results" / "task2_results_mitchell"
OUTPUT_DIR = REPO_ROOT / "results" / "task3_results"

VERSIONS = ["bare", "stop", "lemm", "lemm_stop"]
N_VALUES = [50, 100, 500, 1000, 5000, "all"]
LAMBDA_VALUES = [1, 9, 99]


def n_value_to_text(n_words):
    return str(n_words)


def label_to_number(label):
    if label == "spam":
        return 1
    return 0


def read_task2_predictions(file_path):
    rows = []

    with file_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            rows.append({
                "version": row["version"],
                "n_words": row["n_words"],
                "fold": int(row["fold"]),
                "filename": row["filename"],
                "actual_label": row["actual_label"],
                "actual_label_number": int(row["actual_label_number"]),
                "task2_predicted_label": row["predicted_label"],
                "task2_predicted_label_number": int(row["predicted_label_number"]),
                "task2_correct": row["correct"],
                "spam_log_score": float(row["spam_log_score"]),
                "ham_log_score": float(row["ham_log_score"]),
                "known_token_count": int(row["known_token_count"]),
                "total_token_count": int(row["total_token_count"]),
            })

    return rows


def apply_cost_sensitive_rule(task2_row, lambda_value):
    spam_log_score = task2_row["spam_log_score"]
    ham_log_score = task2_row["ham_log_score"]

    log_ratio = spam_log_score - ham_log_score
    log_threshold = math.log(lambda_value)

    if log_ratio > log_threshold:
        predicted_label = "spam"
    else:
        predicted_label = "ham"

    predicted_label_number = label_to_number(predicted_label)
    actual_label = task2_row["actual_label"]
    actual_label_number = task2_row["actual_label_number"]

    correct = predicted_label_number == actual_label_number

    return {
        "version": task2_row["version"],
        "n_words": task2_row["n_words"],
        "lambda": lambda_value,
        "fold": task2_row["fold"],
        "filename": task2_row["filename"],
        "actual_label": actual_label,
        "actual_label_number": actual_label_number,
        "task2_predicted_label": task2_row["task2_predicted_label"],
        "task2_predicted_label_number": task2_row["task2_predicted_label_number"],
        "cost_sensitive_predicted_label": predicted_label,
        "cost_sensitive_predicted_label_number": predicted_label_number,
        "correct": correct,
        "spam_log_score": spam_log_score,
        "ham_log_score": ham_log_score,
        "log_ratio": log_ratio,
        "log_threshold": log_threshold,
        "known_token_count": task2_row["known_token_count"],
        "total_token_count": task2_row["total_token_count"],
    }


def calculate_metrics(prediction_rows, lambda_value):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for row in prediction_rows:
        actual = row["actual_label"]
        predicted = row["cost_sensitive_predicted_label"]

        if actual == "spam" and predicted == "spam":
            true_positive += 1
        elif actual == "ham" and predicted == "ham":
            true_negative += 1
        elif actual == "ham" and predicted == "spam":
            false_positive += 1
        elif actual == "spam" and predicted == "ham":
            false_negative += 1

    total = len(prediction_rows)
    correct = true_positive + true_negative

    number_spam = true_positive + false_negative
    number_ham = true_negative + false_positive

    accuracy = correct / total if total > 0 else 0

    spam_precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0
    )

    spam_recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0
    )

    weighted_accuracy_denominator = (
        lambda_value * (true_negative + false_positive)
        + (true_positive + false_negative)
    )

    weighted_accuracy = (
        ((lambda_value * true_negative) + true_positive)
        / weighted_accuracy_denominator
        if weighted_accuracy_denominator > 0
        else 0
    )

    baseline_weighted_accuracy = (
        (lambda_value * number_ham)
        / ((lambda_value * number_ham) + number_spam)
        if ((lambda_value * number_ham) + number_spam) > 0
        else 0
    )

    tcr_denominator = false_negative + (lambda_value * false_positive)

    if tcr_denominator == 0:
        tcr = "inf"
    else:
        tcr = number_spam / tcr_denominator

    return {
        "testing_documents": total,
        "number_spam": number_spam,
        "number_ham": number_ham,
        "correct": correct,
        "accuracy": accuracy,
        "TP": true_positive,
        "TN": true_negative,
        "FP": false_positive,
        "FN": false_negative,
        "spam_precision": spam_precision,
        "spam_recall": spam_recall,
        "weighted_accuracy": weighted_accuracy,
        "baseline_weighted_accuracy": baseline_weighted_accuracy,
        "TCR": tcr,
    }


def write_version_predictions(version_output_dir, prediction_rows):
    output_path = version_output_dir / "cost_sensitive_predictions.csv"

    fieldnames = [
        "version",
        "n_words",
        "lambda",
        "fold",
        "filename",
        "actual_label",
        "actual_label_number",
        "task2_predicted_label",
        "task2_predicted_label_number",
        "cost_sensitive_predicted_label",
        "cost_sensitive_predicted_label_number",
        "correct",
        "spam_log_score",
        "ham_log_score",
        "log_ratio",
        "log_threshold",
        "known_token_count",
        "total_token_count",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(prediction_rows)

    return output_path


def write_version_fold_metrics(version_output_dir, metric_rows):
    output_path = version_output_dir / "cost_sensitive_fold_metrics.csv"

    fieldnames = [
        "version",
        "n_words",
        "lambda",
        "fold",
        "testing_documents",
        "number_spam",
        "number_ham",
        "correct",
        "accuracy",
        "TP",
        "TN",
        "FP",
        "FN",
        "spam_precision",
        "spam_recall",
        "weighted_accuracy",
        "baseline_weighted_accuracy",
        "TCR",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)

    return output_path


def write_combined_fold_metrics(all_metric_rows):
    output_path = OUTPUT_DIR / "combined_cost_sensitive_fold_metrics.csv"

    fieldnames = [
        "version",
        "n_words",
        "lambda",
        "fold",
        "testing_documents",
        "number_spam",
        "number_ham",
        "correct",
        "accuracy",
        "TP",
        "TN",
        "FP",
        "FN",
        "spam_precision",
        "spam_recall",
        "weighted_accuracy",
        "baseline_weighted_accuracy",
        "TCR",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_metric_rows)

    return output_path


def create_overall_rows(all_metric_rows):
    overall_rows = []

    for version in VERSIONS:
        for n_words in N_VALUES:
            for lambda_value in LAMBDA_VALUES:
                matching_rows = [
                    row for row in all_metric_rows
                    if row["version"] == version
                    and row["n_words"] == n_words
                    and row["lambda"] == lambda_value
                ]

                total_testing = sum(row["testing_documents"] for row in matching_rows)
                total_spam = sum(row["number_spam"] for row in matching_rows)
                total_ham = sum(row["number_ham"] for row in matching_rows)
                total_correct = sum(row["correct"] for row in matching_rows)

                total_tp = sum(row["TP"] for row in matching_rows)
                total_tn = sum(row["TN"] for row in matching_rows)
                total_fp = sum(row["FP"] for row in matching_rows)
                total_fn = sum(row["FN"] for row in matching_rows)

                accuracy = total_correct / total_testing if total_testing > 0 else 0

                spam_precision = (
                    total_tp / (total_tp + total_fp)
                    if (total_tp + total_fp) > 0
                    else 0
                )

                spam_recall = (
                    total_tp / (total_tp + total_fn)
                    if (total_tp + total_fn) > 0
                    else 0
                )

                weighted_accuracy_denominator = (
                    lambda_value * (total_tn + total_fp)
                    + (total_tp + total_fn)
                )

                weighted_accuracy = (
                    ((lambda_value * total_tn) + total_tp)
                    / weighted_accuracy_denominator
                    if weighted_accuracy_denominator > 0
                    else 0
                )

                baseline_weighted_accuracy = (
                    (lambda_value * total_ham)
                    / ((lambda_value * total_ham) + total_spam)
                    if ((lambda_value * total_ham) + total_spam) > 0
                    else 0
                )

                tcr_denominator = total_fn + (lambda_value * total_fp)

                if tcr_denominator == 0:
                    tcr = "inf"
                else:
                    tcr = total_spam / tcr_denominator

                overall_rows.append({
                    "version": version,
                    "n_words": n_words,
                    "lambda": lambda_value,
                    "testing_documents": total_testing,
                    "number_spam": total_spam,
                    "number_ham": total_ham,
                    "correct": total_correct,
                    "accuracy": accuracy,
                    "TP": total_tp,
                    "TN": total_tn,
                    "FP": total_fp,
                    "FN": total_fn,
                    "spam_precision": spam_precision,
                    "spam_recall": spam_recall,
                    "weighted_accuracy": weighted_accuracy,
                    "baseline_weighted_accuracy": baseline_weighted_accuracy,
                    "TCR": tcr,
                })

    return overall_rows


def write_overall_metrics(overall_rows):
    output_path = OUTPUT_DIR / "overall_task3_metrics.csv"

    fieldnames = [
        "version",
        "n_words",
        "lambda",
        "testing_documents",
        "number_spam",
        "number_ham",
        "correct",
        "accuracy",
        "TP",
        "TN",
        "FP",
        "FN",
        "spam_precision",
        "spam_recall",
        "weighted_accuracy",
        "baseline_weighted_accuracy",
        "TCR",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(overall_rows)

    return output_path


def write_readme():
    output_path = OUTPUT_DIR / "README_task3_outputs.txt"

    with output_path.open("w", encoding="utf-8") as file:
        file.write("Task 3 Metrics Implementation\n")
        file.write("\n")
        file.write("This script reads Task 2 prediction files created by the Mitchell Naive Bayes script.\n")
        file.write("It applies the cost-sensitive decision rule required in the assignment.\n")
        file.write("\n")
        file.write("Decision rule:\n")
        file.write("Classify as spam only if P(Spam | Email) / P(Ham | Email) > lambda.\n")
        file.write("\n")
        file.write("Because Task 2 stored log scores, the script uses:\n")
        file.write("spam_log_score - ham_log_score > log(lambda).\n")
        file.write("\n")
        file.write("Lambda values used:\n")
        file.write("1\n")
        file.write("9\n")
        file.write("99\n")
        file.write("\n")
        file.write("Metrics calculated:\n")
        file.write("Accuracy\n")
        file.write("Spam Precision\n")
        file.write("Spam Recall\n")
        file.write("Weighted Accuracy\n")
        file.write("Baseline Weighted Accuracy\n")
        file.write("Total Cost Ratio\n")
        file.write("TP, TN, FP, FN\n")
        file.write("\n")
        file.write("Main file to open first:\n")
        file.write("overall_task3_metrics.csv\n")

    return output_path


def process_version(version):
    print(f"Processing Task 3 metrics for version: {version}")

    version_task2_dir = TASK2_RESULTS_DIR / version
    version_output_dir = OUTPUT_DIR / version
    version_output_dir.mkdir(parents=True, exist_ok=True)

    if not version_task2_dir.exists():
        raise FileNotFoundError(
            f"Task 2 results folder not found for version {version}: {version_task2_dir}"
        )

    version_prediction_rows = []
    version_metric_rows = []

    for n_words in N_VALUES:
        n_text = n_value_to_text(n_words)

        for fold in range(1, 11):
            task2_file = version_task2_dir / f"predictions_fold{fold}_N{n_text}.csv"

            if not task2_file.exists():
                raise FileNotFoundError(f"Missing Task 2 prediction file: {task2_file}")

            task2_rows = read_task2_predictions(task2_file)

            for lambda_value in LAMBDA_VALUES:
                cost_sensitive_rows = [
                    apply_cost_sensitive_rule(row, lambda_value)
                    for row in task2_rows
                ]

                metrics = calculate_metrics(cost_sensitive_rows, lambda_value)

                metric_row = {
                    "version": version,
                    "n_words": n_words,
                    "lambda": lambda_value,
                    "fold": fold,
                    "testing_documents": metrics["testing_documents"],
                    "number_spam": metrics["number_spam"],
                    "number_ham": metrics["number_ham"],
                    "correct": metrics["correct"],
                    "accuracy": metrics["accuracy"],
                    "TP": metrics["TP"],
                    "TN": metrics["TN"],
                    "FP": metrics["FP"],
                    "FN": metrics["FN"],
                    "spam_precision": metrics["spam_precision"],
                    "spam_recall": metrics["spam_recall"],
                    "weighted_accuracy": metrics["weighted_accuracy"],
                    "baseline_weighted_accuracy": metrics["baseline_weighted_accuracy"],
                    "TCR": metrics["TCR"],
                }

                version_metric_rows.append(metric_row)
                version_prediction_rows.extend(cost_sensitive_rows)

                print(
                    f"  N={n_words}, fold={fold}, lambda={lambda_value}: "
                    f"WAcc={metrics['weighted_accuracy']:.4f}, "
                    f"TCR={metrics['TCR']}, "
                    f"TP={metrics['TP']}, TN={metrics['TN']}, "
                    f"FP={metrics['FP']}, FN={metrics['FN']}"
                )

    predictions_path = write_version_predictions(
        version_output_dir,
        version_prediction_rows,
    )

    metrics_path = write_version_fold_metrics(
        version_output_dir,
        version_metric_rows,
    )

    print(f"Finished version: {version}")
    print(f"  Predictions saved to: {predictions_path}")
    print(f"  Fold metrics saved to: {metrics_path}")
    print()

    return version_metric_rows


def main():
    if not TASK2_RESULTS_DIR.exists():
        raise FileNotFoundError(
            f"Task 2 results folder not found: {TASK2_RESULTS_DIR}. "
            "Run task2_naive_bayes.py first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_metric_rows = []

    for version in VERSIONS:
        version_metric_rows = process_version(version)
        all_metric_rows.extend(version_metric_rows)

    combined_fold_metrics_path = write_combined_fold_metrics(all_metric_rows)

    overall_rows = create_overall_rows(all_metric_rows)
    overall_metrics_path = write_overall_metrics(overall_rows)

    readme_path = write_readme()

    print("Task 3 completed successfully.")
    print()
    print("Results saved in:")
    print(OUTPUT_DIR)
    print()
    print("Open this file first:")
    print(overall_metrics_path)
    print()
    print("Other useful files:")
    print(combined_fold_metrics_path)
    print(readme_path)


if __name__ == "__main__":
    main()