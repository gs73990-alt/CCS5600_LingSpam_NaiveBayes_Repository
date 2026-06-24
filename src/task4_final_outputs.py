from pathlib import Path
import csv
import math

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
except ModuleNotFoundError:
    print("Missing dependency: matplotlib")
    print("Install it by running:")
    print("python -m pip install matplotlib")
    raise


REPO_ROOT = Path(__file__).resolve().parents[1]

TASK3_METRICS_PATH = REPO_ROOT / "results" / "task3_results" / "overall_task3_metrics.csv"
TRAINING_SIZE_TCR_PATH = REPO_ROOT / "results" / "task4_training_size_results" / "learning_curve_tcr_graph_data.csv"
TRAINING_SIZE_WACC_PATH = REPO_ROOT / "results" / "task4_training_size_results" / "learning_curve_wacc_graph_data.csv"

OUTPUT_DIR = REPO_ROOT / "results" / "task4_python_outputs"

ATTRIBUTE_GRAPH_VERSION = "lemm_stop"
COMPARISON_TABLE_N = "all"

VERSIONS = ["bare", "stop", "lemm", "lemm_stop"]
VERSION_DISPLAY_NAMES = {
    "bare": "Bare",
    "stop": "Stop",
    "lemm": "Lemm",
    "lemm_stop": "Lemm_Stop",
}

LAMBDA_VALUES = [1, 9, 99]
N_ORDER = ["50", "100", "500", "1000", "5000", "all"]


def parse_float(value):
    if value == "inf":
        return math.inf
    return float(value)


def format_percent(value):
    return f"{value * 100:.2f}%"


def format_number(value):
    if value == math.inf:
        return "inf"
    return f"{value:.2f}"


def read_task3_metrics():
    if not TASK3_METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Task 3 metrics file not found: {TASK3_METRICS_PATH}\n"
            "Run task3_metrics.py first."
        )

    rows = []

    with TASK3_METRICS_PATH.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        required_columns = {
            "version",
            "n_words",
            "lambda",
            "spam_precision",
            "spam_recall",
            "weighted_accuracy",
            "TCR",
        }

        missing_columns = required_columns - set(reader.fieldnames or [])

        if missing_columns:
            raise ValueError(
                f"Task 3 metrics CSV is missing columns: {sorted(missing_columns)}\n"
                "Make sure overall_task3_metrics.csv is the raw Task 3 output, not a saved PivotTable."
            )

        for row in reader:
            rows.append({
                "version": row["version"],
                "n_words": row["n_words"],
                "lambda": int(row["lambda"]),
                "spam_precision": parse_float(row["spam_precision"]),
                "spam_recall": parse_float(row["spam_recall"]),
                "weighted_accuracy": parse_float(row["weighted_accuracy"]),
                "TCR": parse_float(row["TCR"]),
            })

    return rows


def get_metric_rows_for_version(rows, version, metric_name):
    selected = [
        row for row in rows
        if row["version"] == version
    ]

    selected = [
        row for row in selected
        if row["n_words"] in N_ORDER
    ]

    selected.sort(
        key=lambda row: (
            row["lambda"],
            N_ORDER.index(row["n_words"]),
        )
    )

    data_by_lambda = {}

    for lambda_value in LAMBDA_VALUES:
        lambda_rows = [
            row for row in selected
            if row["lambda"] == lambda_value
        ]

        values = []

        for n_words in N_ORDER:
            matching = [
                row for row in lambda_rows
                if row["n_words"] == n_words
            ]

            if not matching:
                raise ValueError(
                    f"Missing row for version={version}, lambda={lambda_value}, n_words={n_words}"
                )

            values.append(matching[0][metric_name])

        data_by_lambda[lambda_value] = values

    return data_by_lambda


def plot_metric_vs_vocabulary(rows, version, metric_name, y_label, title, output_filename, is_percent=False):
    data_by_lambda = get_metric_rows_for_version(rows, version, metric_name)

    x_positions = list(range(len(N_ORDER)))

    plt.figure(figsize=(9, 5.5), dpi=160)

    colors = {
        1: "#1f77b4",
        9: "#ff7f0e",
        99: "#7f7f7f",
    }

    for lambda_value in LAMBDA_VALUES:
        plt.plot(
            x_positions,
            data_by_lambda[lambda_value],
            marker="o",
            linewidth=2.2,
            markersize=5,
            color=colors[lambda_value],
            label=f"lambda = {lambda_value}",
        )

    plt.title(title, fontsize=14, pad=14)
    plt.xlabel("Vocabulary Size (N)", fontsize=11)
    plt.ylabel(y_label, fontsize=11)
    plt.xticks(x_positions, N_ORDER)
    plt.grid(True, axis="y", linestyle="--", alpha=0.35)
    plt.legend(frameon=False)

    if is_percent:
        all_values = []
        for lambda_value in LAMBDA_VALUES:
            all_values.extend(data_by_lambda[lambda_value])

        lower = max(0, min(all_values) - 0.01)
        upper = min(1.01, max(all_values) + 0.01)

        plt.ylim(lower, upper)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    plt.tight_layout()

    output_path = OUTPUT_DIR / output_filename
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    return output_path


def get_row(rows, version, n_words, lambda_value):
    matching = [
        row for row in rows
        if row["version"] == version
        and row["n_words"] == n_words
        and row["lambda"] == lambda_value
    ]

    if not matching:
        raise ValueError(
            f"Missing comparison row for version={version}, n_words={n_words}, lambda={lambda_value}"
        )

    return matching[0]


def create_preprocessing_comparison_rows(rows):
    table_rows = []

    for version in VERSIONS:
        row_lambda_1 = get_row(rows, version, COMPARISON_TABLE_N, 1)
        row_lambda_9 = get_row(rows, version, COMPARISON_TABLE_N, 9)
        row_lambda_99 = get_row(rows, version, COMPARISON_TABLE_N, 99)

        table_rows.append({
            "Version": VERSION_DISPLAY_NAMES[version],
            "N": COMPARISON_TABLE_N,
            "Precision": format_percent(row_lambda_9["spam_precision"]),
            "Recall": format_percent(row_lambda_9["spam_recall"]),
            "WAcc1": format_percent(row_lambda_1["weighted_accuracy"]),
            "WAcc9": format_percent(row_lambda_9["weighted_accuracy"]),
            "WAcc99": format_percent(row_lambda_99["weighted_accuracy"]),
            "TCR9": format_number(row_lambda_9["TCR"]),
            "TCR99": format_number(row_lambda_99["TCR"]),
        })

    return table_rows


def write_comparison_table_csv(table_rows):
    output_path = OUTPUT_DIR / "preprocessing_comparison_table.csv"

    fieldnames = [
        "Version",
        "N",
        "Precision",
        "Recall",
        "WAcc1",
        "WAcc9",
        "WAcc99",
        "TCR9",
        "TCR99",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table_rows)

    return output_path


def plot_comparison_table(table_rows):
    output_path = OUTPUT_DIR / "preprocessing_comparison_table.png"

    columns = [
        "Version",
        "N",
        "Precision",
        "Recall",
        "WAcc1",
        "WAcc9",
        "WAcc99",
        "TCR9",
        "TCR99",
    ]

    cell_text = []

    for row in table_rows:
        cell_text.append([row[column] for column in columns])

    fig, ax = plt.subplots(figsize=(12, 2.7), dpi=180)
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.45)

    for (row_index, col_index), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d0d0")

        if row_index == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#2f5f8f")
        else:
            if row_index % 2 == 0:
                cell.set_facecolor("#f4f7fb")
            else:
                cell.set_facecolor("white")

    plt.title(
        "Comparative Performance Across Preprocessing Versions (N = all)",
        fontsize=13,
        pad=12,
    )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    return output_path


def read_training_size_tcr_graph_data():
    if not TRAINING_SIZE_TCR_PATH.exists():
        raise FileNotFoundError(
            f"Training-size TCR file not found: {TRAINING_SIZE_TCR_PATH}\n"
            "Run task4_training_size_experiment.py first."
        )

    rows = []

    with TRAINING_SIZE_TCR_PATH.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        required_columns = {
            "training_size_percent",
            "TCR_lambda_1",
            "TCR_lambda_9",
            "TCR_lambda_99",
        }

        missing_columns = required_columns - set(reader.fieldnames or [])

        if missing_columns:
            raise ValueError(
                f"Learning curve TCR CSV is missing columns: {sorted(missing_columns)}"
            )

        for row in reader:
            rows.append({
                "training_size_percent": int(row["training_size_percent"]),
                "TCR_lambda_1": parse_float(row["TCR_lambda_1"]),
                "TCR_lambda_9": parse_float(row["TCR_lambda_9"]),
                "TCR_lambda_99": parse_float(row["TCR_lambda_99"]),
            })

    rows.sort(key=lambda row: row["training_size_percent"])
    return rows


def plot_training_size_tcr(rows):
    output_path = OUTPUT_DIR / "training_size_vs_tcr.png"

    x_values = [row["training_size_percent"] for row in rows]

    series = {
        1: [row["TCR_lambda_1"] for row in rows],
        9: [row["TCR_lambda_9"] for row in rows],
        99: [row["TCR_lambda_99"] for row in rows],
    }

    colors = {
        1: "#ff7f0e",
        9: "#7f7f7f",
        99: "#f2b701",
    }

    plt.figure(figsize=(9, 5.5), dpi=160)

    for lambda_value in LAMBDA_VALUES:
        plt.plot(
            x_values,
            series[lambda_value],
            marker="o",
            linewidth=2.2,
            markersize=5,
            color=colors[lambda_value],
            label=f"lambda = {lambda_value}",
        )

    plt.title("Training Size vs TCR", fontsize=14, pad=14)
    plt.xlabel("Training Size (%)", fontsize=11)
    plt.ylabel("TCR", fontsize=11)
    plt.xticks(x_values)
    plt.grid(True, axis="y", linestyle="--", alpha=0.35)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    return output_path


def write_summary_file(created_files):
    output_path = OUTPUT_DIR / "README_task4_python_outputs.txt"

    with output_path.open("w", encoding="utf-8") as file:
        file.write("Task 4 Python-generated final outputs\n")
        file.write("\n")
        file.write("Source files used:\n")
        file.write(f"{TASK3_METRICS_PATH}\n")
        file.write(f"{TRAINING_SIZE_TCR_PATH}\n")
        file.write("\n")
        file.write("Generated files:\n")

        for path in created_files:
            file.write(f"{path.name}\n")

        file.write("\n")
        file.write("Recommended report usage:\n")
        file.write("1. Use weighted_accuracy_vs_vocab.png for the Weighted Accuracy vs Vocabulary Size graph.\n")
        file.write("2. Use tcr_vs_vocab.png for the TCR vs Vocabulary Size graph.\n")
        file.write("3. Use preprocessing_comparison_table.png or preprocessing_comparison_table.csv for the preprocessing comparison table.\n")
        file.write("4. Use training_size_vs_tcr.png for the learning curve graph.\n")

    return output_path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    task3_rows = read_task3_metrics()

    created_files = []

    weighted_accuracy_graph = plot_metric_vs_vocabulary(
        rows=task3_rows,
        version=ATTRIBUTE_GRAPH_VERSION,
        metric_name="weighted_accuracy",
        y_label="Weighted Accuracy",
        title=f"Weighted Accuracy vs Vocabulary Size ({VERSION_DISPLAY_NAMES[ATTRIBUTE_GRAPH_VERSION]})",
        output_filename="weighted_accuracy_vs_vocab.png",
        is_percent=True,
    )
    created_files.append(weighted_accuracy_graph)

    tcr_graph = plot_metric_vs_vocabulary(
        rows=task3_rows,
        version=ATTRIBUTE_GRAPH_VERSION,
        metric_name="TCR",
        y_label="TCR",
        title=f"TCR vs Vocabulary Size ({VERSION_DISPLAY_NAMES[ATTRIBUTE_GRAPH_VERSION]})",
        output_filename="tcr_vs_vocab.png",
        is_percent=False,
    )
    created_files.append(tcr_graph)

    comparison_rows = create_preprocessing_comparison_rows(task3_rows)

    comparison_csv = write_comparison_table_csv(comparison_rows)
    created_files.append(comparison_csv)

    comparison_png = plot_comparison_table(comparison_rows)
    created_files.append(comparison_png)

    training_size_rows = read_training_size_tcr_graph_data()

    training_curve_graph = plot_training_size_tcr(training_size_rows)
    created_files.append(training_curve_graph)

    summary_file = write_summary_file(created_files)
    created_files.append(summary_file)

    print("Task 4 Python outputs generated successfully.")
    print()
    print("Output folder:")
    print(OUTPUT_DIR)
    print()
    print("Generated files:")

    for path in created_files:
        print(path)


if __name__ == "__main__":
    main()
