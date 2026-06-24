from pathlib import Path
import csv
import math
import re
from collections import Counter

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = REPO_ROOT / "data" / "lingspam_public"
OUTPUT_DIR = REPO_ROOT / "results" / "task2_results_mitchell"

VERSIONS = ["bare", "stop", "lemm", "lemm_stop"]

VERSION_DESCRIPTIONS = {
    "bare": "No lemmatization, stop-words included",
    "stop": "No lemmatization, stop-words removed",
    "lemm": "Lemmatized, stop-words included",
    "lemm_stop": "Lemmatized, stop-words removed",
}

N_VALUES = [50, 100, 500, 1000, 5000, "all"]
CLASSES = ["spam", "ham"]


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
                "version": version_name,
                "filename": file_path.name,
                "path": str(file_path),
                "fold": fold,
                "label": label,
                "label_number": label_to_number(label),
                "subject": subject,
                "body": body,
                "tokens": tokens,
                "token_count": len(tokens),
            })

    return emails


def build_vocabulary(training_emails, n_words):
    word_counts = Counter()

    for email in training_emails:
        word_counts.update(email["tokens"])

    if n_words == "all":
        most_common_words = word_counts.most_common()
    else:
        most_common_words = word_counts.most_common(n_words)

    vocabulary = [word for word, count in most_common_words]
    return vocabulary, word_counts


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

    word_probabilities = {
        "spam": {},
        "ham": {},
    }

    word_log_probabilities = {
        "spam": {},
        "ham": {},
    }

    for class_label in CLASSES:
        denominator = total_word_positions_by_class[class_label] + vocabulary_size

        for word in vocabulary:
            word_count = word_counts_by_class[class_label][word]

            probability = (word_count + 1) / denominator

            word_probabilities[class_label][word] = probability
            word_log_probabilities[class_label][word] = math.log(probability)

    model = {
        "vocabulary": vocabulary,
        "vocabulary_set": vocabulary_set,
        "vocabulary_size": vocabulary_size,
        "total_documents": total_documents,
        "class_document_counts": class_document_counts,
        "priors": priors,
        "word_counts_by_class": word_counts_by_class,
        "total_word_positions_by_class": total_word_positions_by_class,
        "word_probabilities": word_probabilities,
        "word_log_probabilities": word_log_probabilities,
    }

    return model


def classify_mitchell_naive_bayes(model, email):
    spam_log_score = math.log(model["priors"]["spam"])
    ham_log_score = math.log(model["priors"]["ham"])

    known_token_count = 0

    for token in email["tokens"]:
        if token in model["vocabulary_set"]:
            known_token_count += 1
            spam_log_score += model["word_log_probabilities"]["spam"][token]
            ham_log_score += model["word_log_probabilities"]["ham"][token]

    if spam_log_score > ham_log_score:
        predicted_label = "spam"
    else:
        predicted_label = "ham"

    return {
        "predicted_label": predicted_label,
        "predicted_label_number": label_to_number(predicted_label),
        "spam_log_score": spam_log_score,
        "ham_log_score": ham_log_score,
        "known_token_count": known_token_count,
        "total_token_count": email["token_count"],
    }


def evaluate_predictions(prediction_rows):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for row in prediction_rows:
        actual = row["actual_label"]
        predicted = row["predicted_label"]

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

    accuracy = correct / total if total > 0 else 0
    spam_precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    spam_recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

    return {
        "testing_documents": total,
        "correct": correct,
        "accuracy": accuracy,
        "TP": true_positive,
        "TN": true_negative,
        "FP": false_positive,
        "FN": false_negative,
        "spam_precision": spam_precision,
        "spam_recall": spam_recall,
    }


def n_value_to_text(n_words):
    return str(n_words)


def write_vocabulary_file(version_output_dir, fold, n_words, vocabulary, training_word_counts):
    n_text = n_value_to_text(n_words)
    output_path = version_output_dir / f"vocabulary_fold{fold}_N{n_text}.csv"

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["rank", "word", "training_frequency"])

        for rank, word in enumerate(vocabulary, start=1):
            writer.writerow([rank, word, training_word_counts[word]])

    return output_path


def write_predictions_file(version_output_dir, fold, n_words, prediction_rows):
    n_text = n_value_to_text(n_words)
    output_path = version_output_dir / f"predictions_fold{fold}_N{n_text}.csv"

    fieldnames = [
        "version",
        "n_words",
        "fold",
        "filename",
        "actual_label",
        "actual_label_number",
        "predicted_label",
        "predicted_label_number",
        "correct",
        "spam_log_score",
        "ham_log_score",
        "known_token_count",
        "total_token_count",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(prediction_rows)

    return output_path


def write_version_fold_summary(version_output_dir, summary_rows):
    output_path = version_output_dir / "naive_bayes_mitchell_fold_results.csv"

    fieldnames = [
        "version",
        "description",
        "n_words",
        "fold",
        "vocabulary_size",
        "training_documents",
        "training_spam",
        "training_ham",
        "testing_documents",
        "correct",
        "accuracy",
        "TP",
        "TN",
        "FP",
        "FN",
        "spam_precision",
        "spam_recall",
        "prior_spam",
        "prior_ham",
        "spam_word_positions",
        "ham_word_positions",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    return output_path


def write_combined_fold_summary(all_summary_rows):
    output_path = OUTPUT_DIR / "combined_naive_bayes_mitchell_fold_results.csv"

    fieldnames = [
        "version",
        "description",
        "n_words",
        "fold",
        "vocabulary_size",
        "training_documents",
        "training_spam",
        "training_ham",
        "testing_documents",
        "correct",
        "accuracy",
        "TP",
        "TN",
        "FP",
        "FN",
        "spam_precision",
        "spam_recall",
        "prior_spam",
        "prior_ham",
        "spam_word_positions",
        "ham_word_positions",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_summary_rows)

    return output_path


def create_overall_rows(all_summary_rows):
    overall_rows = []

    for version in VERSIONS:
        for n_words in N_VALUES:
            matching_rows = [
                row for row in all_summary_rows
                if row["version"] == version and row["n_words"] == n_words
            ]

            total_testing = sum(row["testing_documents"] for row in matching_rows)
            total_correct = sum(row["correct"] for row in matching_rows)
            total_tp = sum(row["TP"] for row in matching_rows)
            total_tn = sum(row["TN"] for row in matching_rows)
            total_fp = sum(row["FP"] for row in matching_rows)
            total_fn = sum(row["FN"] for row in matching_rows)

            accuracy = total_correct / total_testing if total_testing > 0 else 0
            spam_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            spam_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

            vocabulary_sizes = [row["vocabulary_size"] for row in matching_rows]
            average_vocabulary_size = sum(vocabulary_sizes) / len(vocabulary_sizes) if vocabulary_sizes else 0

            overall_rows.append({
                "version": version,
                "description": VERSION_DESCRIPTIONS[version],
                "n_words": n_words,
                "average_vocabulary_size": average_vocabulary_size,
                "total_testing_documents": total_testing,
                "total_correct": total_correct,
                "accuracy": accuracy,
                "TP": total_tp,
                "TN": total_tn,
                "FP": total_fp,
                "FN": total_fn,
                "spam_precision": spam_precision,
                "spam_recall": spam_recall,
            })

    return overall_rows


def write_overall_summary(overall_rows):
    output_path = OUTPUT_DIR / "overall_naive_bayes_mitchell_results.csv"

    fieldnames = [
        "version",
        "description",
        "n_words",
        "average_vocabulary_size",
        "total_testing_documents",
        "total_correct",
        "accuracy",
        "TP",
        "TN",
        "FP",
        "FN",
        "spam_precision",
        "spam_recall",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(overall_rows)

    return output_path


def write_readme():
    output_path = OUTPUT_DIR / "README_task2_mitchell_outputs.txt"

    with output_path.open("w", encoding="utf-8") as file:
        file.write("Task 2 Naive Bayes results using Tom M. Mitchell text algorithm\n")
        file.write("\n")
        file.write("This implementation trains a Naive Bayes text classifier from scratch.\n")
        file.write("It uses word occurrence counts, class priors, and Laplace smoothing.\n")
        file.write("\n")
        file.write("Mitchell-style training used here:\n")
        file.write("P(v_j) = number of documents in class v_j / total training documents\n")
        file.write("P(w_k | v_j) = (count of word w_k in class v_j + 1) / (total word positions in class v_j + vocabulary size)\n")
        file.write("\n")
        file.write("Classification uses log probabilities to avoid numerical underflow.\n")
        file.write("Only words that appear in the training vocabulary are used during classification.\n")
        file.write("\n")
        file.write("Processed corpus versions:\n")

        for version in VERSIONS:
            file.write(f"{version}: {VERSION_DESCRIPTIONS[version]}\n")

        file.write("\n")
        file.write("Vocabulary sizes used:\n")

        for n_words in N_VALUES:
            file.write(f"N = {n_words}\n")

        file.write("\n")
        file.write("Main files:\n")
        file.write("overall_naive_bayes_mitchell_results.csv: best file to inspect first.\n")
        file.write("combined_naive_bayes_mitchell_fold_results.csv: fold-by-fold results for every version and N value.\n")
        file.write("Each version folder contains prediction files and vocabulary files.\n")

    return output_path


def process_version(version):
    print(f"Processing version: {version}")

    version_output_dir = OUTPUT_DIR / version
    version_output_dir.mkdir(parents=True, exist_ok=True)

    emails = load_dataset(version)
    version_summary_rows = []

    total_spam = sum(1 for email in emails if email["label"] == "spam")
    total_ham = sum(1 for email in emails if email["label"] == "ham")

    print(f"  Total emails: {len(emails)}")
    print(f"  Spam emails: {total_spam}")
    print(f"  Ham emails: {total_ham}")

    for n_words in N_VALUES:
        print(f"  Vocabulary size setting N={n_words}")

        for fold in range(1, 11):
            training_emails = [email for email in emails if email["fold"] != fold]
            testing_emails = [email for email in emails if email["fold"] == fold]

            vocabulary, training_word_counts = build_vocabulary(training_emails, n_words)
            model = train_mitchell_naive_bayes(training_emails, vocabulary)

            prediction_rows = []

            for email in testing_emails:
                prediction = classify_mitchell_naive_bayes(model, email)

                actual_label = email["label"]
                predicted_label = prediction["predicted_label"]
                correct = actual_label == predicted_label

                prediction_rows.append({
                    "version": version,
                    "n_words": n_words,
                    "fold": fold,
                    "filename": email["filename"],
                    "actual_label": actual_label,
                    "actual_label_number": email["label_number"],
                    "predicted_label": predicted_label,
                    "predicted_label_number": prediction["predicted_label_number"],
                    "correct": correct,
                    "spam_log_score": prediction["spam_log_score"],
                    "ham_log_score": prediction["ham_log_score"],
                    "known_token_count": prediction["known_token_count"],
                    "total_token_count": prediction["total_token_count"],
                })

            evaluation = evaluate_predictions(prediction_rows)

            summary_row = {
                "version": version,
                "description": VERSION_DESCRIPTIONS[version],
                "n_words": n_words,
                "fold": fold,
                "vocabulary_size": model["vocabulary_size"],
                "training_documents": model["total_documents"],
                "training_spam": model["class_document_counts"]["spam"],
                "training_ham": model["class_document_counts"]["ham"],
                "testing_documents": evaluation["testing_documents"],
                "correct": evaluation["correct"],
                "accuracy": evaluation["accuracy"],
                "TP": evaluation["TP"],
                "TN": evaluation["TN"],
                "FP": evaluation["FP"],
                "FN": evaluation["FN"],
                "spam_precision": evaluation["spam_precision"],
                "spam_recall": evaluation["spam_recall"],
                "prior_spam": model["priors"]["spam"],
                "prior_ham": model["priors"]["ham"],
                "spam_word_positions": model["total_word_positions_by_class"]["spam"],
                "ham_word_positions": model["total_word_positions_by_class"]["ham"],
            }

            version_summary_rows.append(summary_row)

            write_vocabulary_file(
                version_output_dir,
                fold,
                n_words,
                vocabulary,
                training_word_counts,
            )

            write_predictions_file(
                version_output_dir,
                fold,
                n_words,
                prediction_rows,
            )

            print(
                f"    Fold {fold}: "
                f"accuracy={evaluation['accuracy']:.4f}, "
                f"TP={evaluation['TP']}, "
                f"TN={evaluation['TN']}, "
                f"FP={evaluation['FP']}, "
                f"FN={evaluation['FN']}"
            )

    version_summary_path = write_version_fold_summary(
        version_output_dir,
        version_summary_rows,
    )

    print(f"  Saved version summary: {version_summary_path}")
    print()

    return version_summary_rows


def main():
    if not CORPUS_ROOT.exists():
        raise FileNotFoundError(f"Corpus root folder not found: {CORPUS_ROOT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_summary_rows = []

    for version in VERSIONS:
        version_summary_rows = process_version(version)
        all_summary_rows.extend(version_summary_rows)

    combined_fold_summary_path = write_combined_fold_summary(all_summary_rows)

    overall_rows = create_overall_rows(all_summary_rows)
    overall_summary_path = write_overall_summary(overall_rows)

    readme_path = write_readme()

    print("Task 2 completed successfully.")
    print()
    print("Results saved in:")
    print(OUTPUT_DIR)
    print()
    print("Open this file first:")
    print(overall_summary_path)
    print()
    print("Other useful files:")
    print(combined_fold_summary_path)
    print(readme_path)
    print()
    print("This script used Tom M. Mitchell's Naive Bayes text method with Laplace smoothing.")
    print("It processed all four corpus versions and all vocabulary sizes.")


if __name__ == "__main__":
    main()