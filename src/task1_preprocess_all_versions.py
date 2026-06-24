from pathlib import Path
import csv
import re
from collections import Counter

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = REPO_ROOT / "data" / "lingspam_public"
OUTPUT_DIR = REPO_ROOT / "results" / "task1_results_all_versions"

VERSIONS = ["bare", "stop", "lemm", "lemm_stop"]

VERSION_DESCRIPTIONS = {
    "bare": "No lemmatization, stop-words included",
    "stop": "No lemmatization, stop-words removed",
    "lemm": "Lemmatized, stop-words included",
    "lemm_stop": "Lemmatized, stop-words removed",
}

N_WORDS_FOR_VECTOR_FILES = 100
N_VALUES_FOR_SUMMARY = [50, 100, 500, 1000, 5000, "all"]


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
        return [word for word, count in word_counts.most_common()]

    return [word for word, count in word_counts.most_common(n_words)]


def vectorize_email_binary(email, vocabulary):
    token_set = set(email["tokens"])
    return [1 if word in token_set else 0 for word in vocabulary]


def write_email_metadata(version_output_dir, emails):
    output_path = version_output_dir / "email_metadata.csv"

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "version",
            "filename",
            "fold",
            "label",
            "label_number",
            "subject",
            "token_count",
            "path",
        ])

        for email in emails:
            writer.writerow([
                email["version"],
                email["filename"],
                email["fold"],
                email["label"],
                email["label_number"],
                email["subject"],
                email["token_count"],
                email["path"],
            ])

    return output_path


def write_fold_summary(version_output_dir, emails):
    output_path = version_output_dir / "fold_summary.csv"

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "version",
            "fold",
            "total_emails",
            "spam_emails",
            "ham_emails",
        ])

        for fold in range(1, 11):
            fold_emails = [email for email in emails if email["fold"] == fold]
            spam_count = sum(1 for email in fold_emails if email["label"] == "spam")
            ham_count = sum(1 for email in fold_emails if email["label"] == "ham")

            writer.writerow([
                emails[0]["version"],
                fold,
                len(fold_emails),
                spam_count,
                ham_count,
            ])

    return output_path


def create_cross_validation_rows(version_name, emails):
    rows = []

    for n_words in N_VALUES_FOR_SUMMARY:
        for test_fold in range(1, 11):
            training_emails = [email for email in emails if email["fold"] != test_fold]
            testing_emails = [email for email in emails if email["fold"] == test_fold]

            vocabulary = build_vocabulary(training_emails, n_words)

            train_spam = sum(1 for email in training_emails if email["label"] == "spam")
            train_ham = len(training_emails) - train_spam

            test_spam = sum(1 for email in testing_emails if email["label"] == "spam")
            test_ham = len(testing_emails) - test_spam

            rows.append({
                "version": version_name,
                "description": VERSION_DESCRIPTIONS[version_name],
                "n_words": n_words,
                "test_fold": test_fold,
                "training_emails": len(training_emails),
                "testing_emails": len(testing_emails),
                "train_spam": train_spam,
                "train_ham": train_ham,
                "test_spam": test_spam,
                "test_ham": test_ham,
                "vocabulary_size": len(vocabulary),
                "first_20_words": " ".join(vocabulary[:20]),
            })

    return rows


def write_cross_validation_summary(version_output_dir, rows):
    output_path = version_output_dir / "cross_validation_summary.csv"

    fieldnames = [
        "version",
        "description",
        "n_words",
        "test_fold",
        "training_emails",
        "testing_emails",
        "train_spam",
        "train_ham",
        "test_spam",
        "test_ham",
        "vocabulary_size",
        "first_20_words",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def write_vocabulary(version_output_dir, test_fold, vocabulary):
    output_path = version_output_dir / f"vocabulary_fold{test_fold}_N{N_WORDS_FOR_VECTOR_FILES}.csv"

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "rank",
            "word",
        ])

        for rank, word in enumerate(vocabulary, start=1):
            writer.writerow([
                rank,
                word,
            ])

    return output_path


def write_vectors(version_output_dir, split_name, test_fold, emails, vectors, vocabulary):
    output_path = version_output_dir / f"{split_name}_binary_vectors_fold{test_fold}_N{N_WORDS_FOR_VECTOR_FILES}.csv"

    feature_headers = [f"word_{index + 1}" for index in range(len(vocabulary))]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "version",
            "filename",
            "fold",
            "label",
            "label_number",
        ] + feature_headers)

        for email, vector in zip(emails, vectors):
            writer.writerow([
                email["version"],
                email["filename"],
                email["fold"],
                email["label"],
                email["label_number"],
            ] + vector)

    return output_path


def write_vector_files_for_all_folds(version_output_dir, emails):
    created_files = []

    for test_fold in range(1, 11):
        training_emails = [email for email in emails if email["fold"] != test_fold]
        testing_emails = [email for email in emails if email["fold"] == test_fold]

        vocabulary = build_vocabulary(training_emails, N_WORDS_FOR_VECTOR_FILES)

        train_vectors = [
            vectorize_email_binary(email, vocabulary)
            for email in training_emails
        ]

        test_vectors = [
            vectorize_email_binary(email, vocabulary)
            for email in testing_emails
        ]

        vocabulary_path = write_vocabulary(version_output_dir, test_fold, vocabulary)

        train_path = write_vectors(
            version_output_dir,
            "train",
            test_fold,
            training_emails,
            train_vectors,
            vocabulary,
        )

        test_path = write_vectors(
            version_output_dir,
            "test",
            test_fold,
            testing_emails,
            test_vectors,
            vocabulary,
        )

        created_files.append(vocabulary_path)
        created_files.append(train_path)
        created_files.append(test_path)

    return created_files


def write_text_report(version_output_dir, version_name, emails):
    output_path = version_output_dir / "task1_report_summary.txt"

    total_emails = len(emails)
    total_spam = sum(1 for email in emails if email["label"] == "spam")
    total_ham = sum(1 for email in emails if email["label"] == "ham")

    with output_path.open("w", encoding="utf-8") as file:
        file.write("Task 1: Data Preprocessing and Baseline Exploration\n")
        file.write(f"Corpus version: {version_name}\n")
        file.write(f"Description: {VERSION_DESCRIPTIONS[version_name]}\n")
        file.write(f"Dataset folder: {CORPUS_ROOT / version_name}\n")
        file.write(f"Total emails: {total_emails}\n")
        file.write(f"Spam emails: {total_spam}\n")
        file.write(f"Ham emails: {total_ham}\n")
        file.write("\n")
        file.write("Preprocessing completed:\n")
        file.write("1. Loaded emails from part1 to part10.\n")
        file.write("2. Assigned labels using filenames.\n")
        file.write("3. Files starting with spmsg were labelled as spam.\n")
        file.write("4. All other files were labelled as ham.\n")
        file.write("5. Extracted subject from the first Subject line.\n")
        file.write("6. Extracted body from the remaining email text.\n")
        file.write("7. Tokenized text into lowercase alphabetic words.\n")
        file.write("8. Built vocabularies using training folds only.\n")
        file.write("9. Created binary feature vectors where 1 means the word appears and 0 means the word does not appear.\n")
        file.write("\n")
        file.write("Label encoding:\n")
        file.write("spam = 1\n")
        file.write("ham = 0\n")

    return output_path


def create_dataset_summary_row(version_name, emails):
    total_spam = sum(1 for email in emails if email["label"] == "spam")
    total_ham = sum(1 for email in emails if email["label"] == "ham")

    return {
        "version": version_name,
        "description": VERSION_DESCRIPTIONS[version_name],
        "total_emails": len(emails),
        "spam_emails": total_spam,
        "ham_emails": total_ham,
    }


def write_combined_dataset_summary(dataset_rows):
    output_path = OUTPUT_DIR / "combined_dataset_summary.csv"

    fieldnames = [
        "version",
        "description",
        "total_emails",
        "spam_emails",
        "ham_emails",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dataset_rows)

    return output_path


def write_combined_cross_validation_summary(all_rows):
    output_path = OUTPUT_DIR / "combined_cross_validation_summary.csv"

    fieldnames = [
        "version",
        "description",
        "n_words",
        "test_fold",
        "training_emails",
        "testing_emails",
        "train_spam",
        "train_ham",
        "test_spam",
        "test_ham",
        "vocabulary_size",
        "first_20_words",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    return output_path


def write_main_readme(dataset_rows):
    output_path = OUTPUT_DIR / "README_task1_outputs.txt"

    with output_path.open("w", encoding="utf-8") as file:
        file.write("Task 1 outputs for all Ling-Spam corpus versions\n")
        file.write("\n")
        file.write("Processed versions:\n")

        for row in dataset_rows:
            file.write(f"{row['version']}: {row['description']}\n")

        file.write("\n")
        file.write("Important files:\n")
        file.write("combined_dataset_summary.csv: Overall email counts for all four versions.\n")
        file.write("combined_cross_validation_summary.csv: Training and testing fold summaries for all versions and vocabulary sizes.\n")
        file.write("Each version folder contains email_metadata.csv, fold_summary.csv, cross_validation_summary.csv, task1_report_summary.txt, vocabulary files, train vectors, and test vectors.\n")
        file.write("\n")
        file.write("Vector files:\n")
        file.write(f"Binary vector files were created for N={N_WORDS_FOR_VECTOR_FILES} words.\n")
        file.write("A value of 1 means the vocabulary word appears in the email.\n")
        file.write("A value of 0 means the vocabulary word does not appear in the email.\n")
        file.write("\n")
        file.write("Label encoding:\n")
        file.write("spam = 1\n")
        file.write("ham = 0\n")

    return output_path


def process_version(version_name):
    print(f"Processing {version_name}...")

    version_output_dir = OUTPUT_DIR / version_name
    version_output_dir.mkdir(parents=True, exist_ok=True)

    emails = load_dataset(version_name)

    metadata_path = write_email_metadata(version_output_dir, emails)
    fold_summary_path = write_fold_summary(version_output_dir, emails)

    cross_validation_rows = create_cross_validation_rows(version_name, emails)
    cross_validation_path = write_cross_validation_summary(
        version_output_dir,
        cross_validation_rows,
    )

    vector_files = write_vector_files_for_all_folds(version_output_dir, emails)
    report_path = write_text_report(version_output_dir, version_name, emails)

    dataset_summary_row = create_dataset_summary_row(version_name, emails)

    print(f"Finished {version_name}")
    print(f"  Total emails: {dataset_summary_row['total_emails']}")
    print(f"  Spam emails: {dataset_summary_row['spam_emails']}")
    print(f"  Ham emails: {dataset_summary_row['ham_emails']}")
    print(f"  Output folder: {version_output_dir}")
    print()

    return {
        "version": version_name,
        "emails": emails,
        "dataset_summary_row": dataset_summary_row,
        "cross_validation_rows": cross_validation_rows,
        "metadata_path": metadata_path,
        "fold_summary_path": fold_summary_path,
        "cross_validation_path": cross_validation_path,
        "report_path": report_path,
        "vector_files": vector_files,
    }


def main():
    if not CORPUS_ROOT.exists():
        raise FileNotFoundError(f"Corpus root folder not found: {CORPUS_ROOT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_dataset_rows = []
    all_cross_validation_rows = []
    processed_results = []

    for version_name in VERSIONS:
        result = process_version(version_name)

        processed_results.append(result)
        all_dataset_rows.append(result["dataset_summary_row"])
        all_cross_validation_rows.extend(result["cross_validation_rows"])

    combined_dataset_path = write_combined_dataset_summary(all_dataset_rows)
    combined_cross_validation_path = write_combined_cross_validation_summary(
        all_cross_validation_rows
    )
    readme_path = write_main_readme(all_dataset_rows)

    print("All Task 1 preprocessing completed successfully.")
    print()
    print("Results saved in:")
    print(OUTPUT_DIR)
    print()
    print("Combined files created:")
    print(combined_dataset_path)
    print(combined_cross_validation_path)
    print(readme_path)
    print()
    print("Version folders created:")

    for result in processed_results:
        version_folder = OUTPUT_DIR / result["version"]
        print(version_folder)

    print()
    print("Check combined_dataset_summary.csv first.")
    print("Each version should show 2893 total emails, 481 spam emails, and 2412 ham emails.")


if __name__ == "__main__":
    main()