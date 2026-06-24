from pathlib import Path
import csv
import math
import random
import re
from collections import Counter, defaultdict

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = REPO_ROOT / "data" / "lingspam_public"
OUTPUT_DIR = REPO_ROOT / "results" / "task4_training_size_results"

BEST_VERSION = "stop"
N_WORDS = "all"

TRAINING_PERCENTAGES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
LAMBDA_VALUES = [1, 9, 99]
RANDOM_SEED = 5600


def read_email(file_path):
    text = file_path.read_text(encoding="latin-1")
    lines = text.splitlines()

    if lines and lines[0].lower().startswith("subject:"):
        subject = lines[0][len("Subject:"):].strip()
        body = "\n".join(lines[1:]).strip()
    else:
        subject = ""
        body = text.strip()

    return subject, body


def tokenize(text):
    return re.findall(r"[a-zA-Z]+", text.lower())


def label_to_number(label):
    if label == "spam":
        return 1
    return 0


def get_part_folders(version_dir):
    part_folders = []

    for folder in version_dir.iterdir():
        if folder.is_dir() and re.fullmatch(r"part\d+", folder.name):
            part_folders.append(folder)

    return sorted(part_folders, key=lambda folder: int(folder.name.replace("part", "")))


def load_dataset(version_name):
    version_dir = CORPUS_ROOT / version_name

    if not version_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {version_dir}")

    emails = []

    for part_folder in get_part_folders(version_dir):
        fold = int(part_folder.name.replace("part", ""))

        for file_path in sorted(part_folder.glob("*.txt")):
            label = "spam" if file_path.name.startswith("spmsg") else "ham"
            subject, body = read_email(file_path)
            full_text = subject + " " + body
            tokens = tokenize(full_text)

            emails.append({
                "filename": file_path.name,
                "path": str(file_path),
                "fold": fold,
                "label": label,
                "label_number": label_to_number(label),
                "tokens": tokens,
                "token_count": len(tokens),
            })

    return emails


def select_training_subset(training_emails, training_percentage):
    emails_by_fold = defaultdict(list)

    for email in training_emails:
        emails_by_fold[email["fold"]].append(email)

    selected_emails = []

    for fold in sorted(emails_by_fold):
        fold_emails = list(emails_by_fold[fold])
        random_generator = random.Random(RANDOM_SEED + fold + training_percentage)
        random_generator.shuffle(fold_emails)

        number_to_keep = round(len(fold_emails) * training_percentage / 100)

        if number_to_keep < 1:
            number_to_keep = 1

        selected_emails.extend(fold_emails[:number_to_keep])

    return selected_emails


def build_vocabulary(training_emails, n_words):
    word_counts = Counter()

    for email in training_emails:
        word_counts.update(email["tokens"])

    if n_words == "all":
        most_common_words = word_counts.most_common()
    else:
        most_common_words = word_counts.most_common(n_words)

    vocabulary = [word for word, count in most_common_words]

    return vocabulary


def train_mitchell_naive_bayes(training_emails, vocabulary):
    vocabulary_set = set(vocabulary)
    vocabulary_size = len(vocabulary)
    total_documents = len(training_emails)

    class_document_counts = {
        "spam": sum(1 for email in training_emails if email["label"] == "spam"),
        "ham": sum(1 for email in training_emails if email["label"] == "ham"),
    }

    priors = {
        "spam": class_document_counts["spam"] / total_documents,
        "ham": class_document_counts["ham"] / total_documents,
    }

    word_counts_by_class = {
        "spam": Counter(),
        "ham": Counter(),
    }

    total_word_positions_by_class = {
        "spam": 0,
        "ham": 0,
    }

    for email in training_emails:
        class_label = email["label"]

        for token in email["tokens"]:
            if token in vocabulary_set:
                word_counts_by_class[class_label][token] += 1
                total_word_positions_by_class[class_label] += 1

    word_log_probabilities = {
        "spam": {},
        "ham": {},
    }

    for class_label in ["spam", "ham"]:
        denominator = total_word_positions_by_class[class_label] + vocabulary_size

        for word in vocabulary:
            word_count = word_counts_by_class[class_label][word]
            probability = (word_count + 1) / denominator
            word_log_probabilities[class_label][word] = math.log(probability)

    return {
        "vocabulary": vocabulary,
        "vocabulary_set": vocabulary_set,
        "vocabulary_size": vocabulary_size,
        "total_documents": total_documents,
        "class_document_counts": class_document_counts,
        "priors": priors,
        "word_log_probabilities": word_log_probabilities,
        "total_word_positions_by_class": total_word_positions_by_class,
    }


def classify_email(model, email, lambda_value):
    spam_log_score = math.log(model["priors"]["spam"])
    ham_log_score = math.log(model["priors"]["ham"])

    for token in email["tokens"]:
        if token in model["vocabulary_set"]:
            spam_log_score += model["word_log_probabilities"]["spam"][token]
            ham_log_score += model["word_log_probabilities"]["ham"][token]

    log_ratio = spam_log_score - ham_log_score
    log_threshold = math.log(lambda_value)

    if log_ratio > log_threshold:
        predicted_label = "spam"
    else:
        predicted_label = "ham"

    return {
        "predicted_label": predicted_label,
        "predicted_label_number": label_to_number(predicted_label),
        "spam_log_score": spam_log_score,
        "ham_log_score": ham_log_score,
        "log_ratio": log_ratio,
        "log_threshold": log_threshold,
    }


def calculate_metrics(prediction_rows, lambda_value):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for row in prediction_rows:
        actual = row["actual_label"]
        predicted = row["predicted_label"]

        if actual == "spam" and predicted == "spam":
            tp += 1
        elif actual == "ham" and predicted == "ham":
            tn += 1
        elif actual == "ham" and predicted == "spam":
            fp += 1
        elif actual == "spam" and predicted == "ham":
            fn += 1

    total = len(prediction_rows)
    correct = tp + tn

    number_spam = tp + fn
    number_ham = tn + fp

    accuracy = correct / total if total > 0 else 0

    spam_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    spam_recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    weighted_accuracy_denominator = (
        lambda_value * (tn + fp)
        + (tp + fn)
    )

    weighted_accuracy = (
        ((lambda_value * tn) + tp) / weighted_accuracy_denominator
        if weighted_accuracy_denominator > 0
        else 0
    )

    tcr_denominator = fn + (lambda_value * fp)

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
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "spam_precision": spam_precision,
        "spam_recall": spam_recall,
        "weighted_accuracy": weighted_accuracy,
        "TCR": tcr,
    }


def write_fold_results(rows):
    output_path = OUTPUT_DIR / "training_size_fold_results.csv"

    fieldnames = [
        "version",
        "n_words",
        "training_size_percent",
        "lambda",
        "fold",
        "training_documents",
        "vocabulary_size",
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
        "TCR",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def create_overall_rows(fold_rows):
    overall_rows = []

    for training_percentage in TRAINING_PERCENTAGES:
        for lambda_value in LAMBDA_VALUES:
            matching_rows = [
                row for row in fold_rows
                if row["training_size_percent"] == training_percentage
                and row["lambda"] == lambda_value
            ]

            testing_documents = sum(row["testing_documents"] for row in matching_rows)
            number_spam = sum(row["number_spam"] for row in matching_rows)
            number_ham = sum(row["number_ham"] for row in matching_rows)
            correct = sum(row["correct"] for row in matching_rows)

            tp = sum(row["TP"] for row in matching_rows)
            tn = sum(row["TN"] for row in matching_rows)
            fp = sum(row["FP"] for row in matching_rows)
            fn = sum(row["FN"] for row in matching_rows)

            accuracy = correct / testing_documents if testing_documents > 0 else 0
            spam_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            spam_recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            weighted_accuracy_denominator = (
                lambda_value * (tn + fp)
                + (tp + fn)
            )

            weighted_accuracy = (
                ((lambda_value * tn) + tp) / weighted_accuracy_denominator
                if weighted_accuracy_denominator > 0
                else 0
            )

            tcr_denominator = fn + (lambda_value * fp)

            if tcr_denominator == 0:
                tcr = "inf"
            else:
                tcr = number_spam / tcr_denominator

            average_training_documents = (
                sum(row["training_documents"] for row in matching_rows) / len(matching_rows)
                if matching_rows
                else 0
            )

            average_vocabulary_size = (
                sum(row["vocabulary_size"] for row in matching_rows) / len(matching_rows)
                if matching_rows
                else 0
            )

            overall_rows.append({
                "version": BEST_VERSION,
                "n_words": N_WORDS,
                "training_size_percent": training_percentage,
                "lambda": lambda_value,
                "average_training_documents": average_training_documents,
                "average_vocabulary_size": average_vocabulary_size,
                "testing_documents": testing_documents,
                "number_spam": number_spam,
                "number_ham": number_ham,
                "correct": correct,
                "accuracy": accuracy,
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "spam_precision": spam_precision,
                "spam_recall": spam_recall,
                "weighted_accuracy": weighted_accuracy,
                "TCR": tcr,
            })

    return overall_rows


def write_overall_results(rows):
    output_path = OUTPUT_DIR / "training_size_overall_results.csv"

    fieldnames = [
        "version",
        "n_words",
        "training_size_percent",
        "lambda",
        "average_training_documents",
        "average_vocabulary_size",
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
        "TCR",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def write_tcr_graph_data(overall_rows):
    output_path = OUTPUT_DIR / "learning_curve_tcr_graph_data.csv"

    rows_by_percentage = {}

    for training_percentage in TRAINING_PERCENTAGES:
        rows_by_percentage[training_percentage] = {
            "training_size_percent": training_percentage,
            "TCR_lambda_1": "",
            "TCR_lambda_9": "",
            "TCR_lambda_99": "",
        }

    for row in overall_rows:
        lambda_value = row["lambda"]
        training_percentage = row["training_size_percent"]
        rows_by_percentage[training_percentage][f"TCR_lambda_{lambda_value}"] = row["TCR"]

    fieldnames = [
        "training_size_percent",
        "TCR_lambda_1",
        "TCR_lambda_9",
        "TCR_lambda_99",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for training_percentage in TRAINING_PERCENTAGES:
            writer.writerow(rows_by_percentage[training_percentage])

    return output_path


def write_wacc_graph_data(overall_rows):
    output_path = OUTPUT_DIR / "learning_curve_wacc_graph_data.csv"

    rows_by_percentage = {}

    for training_percentage in TRAINING_PERCENTAGES:
        rows_by_percentage[training_percentage] = {
            "training_size_percent": training_percentage,
            "WAcc_lambda_1": "",
            "WAcc_lambda_9": "",
            "WAcc_lambda_99": "",
        }

    for row in overall_rows:
        lambda_value = row["lambda"]
        training_percentage = row["training_size_percent"]
        rows_by_percentage[training_percentage][f"WAcc_lambda_{lambda_value}"] = row["weighted_accuracy"]

    fieldnames = [
        "training_size_percent",
        "WAcc_lambda_1",
        "WAcc_lambda_9",
        "WAcc_lambda_99",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for training_percentage in TRAINING_PERCENTAGES:
            writer.writerow(rows_by_percentage[training_percentage])

    return output_path


def write_readme():
    output_path = OUTPUT_DIR / "README_training_size_experiment.txt"

    with output_path.open("w", encoding="utf-8") as file:
        file.write("Task 4 Training-Size Experiment\n")
        file.write("\n")
        file.write(f"Preprocessing version used: {BEST_VERSION}\n")
        file.write(f"Vocabulary size setting used: {N_WORDS}\n")
        file.write("\n")
        file.write("Training sizes tested:\n")
        file.write("10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 100%\n")
        file.write("\n")
        file.write("Lambda values tested:\n")
        file.write("1, 9, 99\n")
        file.write("\n")
        file.write("Main files:\n")
        file.write("training_size_overall_results.csv\n")
        file.write("learning_curve_tcr_graph_data.csv\n")
        file.write("learning_curve_wacc_graph_data.csv\n")
        file.write("\n")
        file.write("Use learning_curve_tcr_graph_data.csv to create the report graph for Training Size vs TCR.\n")

    return output_path


def run_experiment():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    emails = load_dataset(BEST_VERSION)

    fold_rows = []

    print("Task 4 training-size experiment")
    print("Version:", BEST_VERSION)
    print("N words:", N_WORDS)
    print("Total emails:", len(emails))
    print()

    for training_percentage in TRAINING_PERCENTAGES:
        print(f"Training size: {training_percentage}%")

        for fold in range(1, 11):
            full_training_emails = [email for email in emails if email["fold"] != fold]
            testing_emails = [email for email in emails if email["fold"] == fold]

            training_subset = select_training_subset(
                full_training_emails,
                training_percentage,
            )

            vocabulary = build_vocabulary(training_subset, N_WORDS)
            model = train_mitchell_naive_bayes(training_subset, vocabulary)

            for lambda_value in LAMBDA_VALUES:
                prediction_rows = []

                for email in testing_emails:
                    prediction = classify_email(model, email, lambda_value)

                    prediction_rows.append({
                        "actual_label": email["label"],
                        "predicted_label": prediction["predicted_label"],
                    })

                metrics = calculate_metrics(prediction_rows, lambda_value)

                fold_rows.append({
                    "version": BEST_VERSION,
                    "n_words": N_WORDS,
                    "training_size_percent": training_percentage,
                    "lambda": lambda_value,
                    "fold": fold,
                    "training_documents": model["total_documents"],
                    "vocabulary_size": model["vocabulary_size"],
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
                    "TCR": metrics["TCR"],
                })

        print(f"  Finished {training_percentage}%")

    fold_results_path = write_fold_results(fold_rows)

    overall_rows = create_overall_rows(fold_rows)
    overall_results_path = write_overall_results(overall_rows)

    tcr_graph_path = write_tcr_graph_data(overall_rows)
    wacc_graph_path = write_wacc_graph_data(overall_rows)

    readme_path = write_readme()

    print()
    print("Training-size experiment completed successfully.")
    print()
    print("Results saved in:")
    print(OUTPUT_DIR)
    print()
    print("Main files:")
    print(fold_results_path)
    print(overall_results_path)
    print(tcr_graph_path)
    print(wacc_graph_path)
    print(readme_path)


if __name__ == "__main__":
    run_experiment()