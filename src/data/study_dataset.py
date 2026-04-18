# ============================================================
#  src/data/study_dataset.py
#  Run from notebook:
#    %run /content/src/data/study_dataset.py
#    study = DatasetStudy()
#    study.run_all()
# ============================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import DATASET_ROOT


class DatasetStudy:
    """
    Analyse and visualise the UCF101 CSV split files.

    Usage
    -----
    study = DatasetStudy()                           # uses DATASET_ROOT from config
    study = DatasetStudy("/custom/path/to/dataset") # override path

    # run everything
    study.run_all()

    # or one plot at a time
    study.plot_class_distribution()
    study.plot_distribution_shape()
    study.plot_box_violin()
    study.plot_top10_pie()
    study.print_balance_score()
    study.print_outliers()
    """

    def __init__(self, dataset_root=DATASET_ROOT):
        self.dataset_root = dataset_root
        self.df = None
        self.class_counts = None
        self._load()

    # Load data
    def _load(self):
        splits = {}
        for split in ["train", "val", "test"]:
            path = os.path.join(self.dataset_root, f"{split}.csv")
            if os.path.exists(path):
                splits[split] = pd.read_csv(path)
                print(f"Loaded {split:5s}: {len(splits[split]):,} rows")
            else:
                print(f"Warning: {path} not found, skipping.")

        if not splits:
            raise FileNotFoundError(f"No CSV files found in {self.dataset_root}")

        self.df = pd.concat(splits.values(), ignore_index=True)

        class_counts = self.df["label"].value_counts().reset_index()
        class_counts.columns = ["Class", "Count"]
        self.class_counts = class_counts.sort_values("Count", ascending=True).reset_index(drop=True)

        print(f"Total videos: {len(self.df):,}")
        print(f"Total classes: {self.class_counts['Class'].nunique()}")

    # Info
    def print_info(self):
        print("\n=== DataFrame Info ===")
        self.df.info()

    # Plot 1
    def plot_class_distribution(self):
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(12, max(10, len(self.class_counts) * 0.22)))

        barplot = sns.barplot(
            data=self.class_counts,
            y="Class",
            x="Count",
            hue="Class",
            palette="viridis",
            legend=False,
            dodge=False,
            orient="h"
        )

        plt.title("UCF101 Class Distribution (Sorted)", fontsize=18, fontweight="bold")
        plt.xlabel("Number of Videos", fontsize=14)
        plt.ylabel("Class Name", fontsize=14)

        for i, v in enumerate(self.class_counts["Count"]):
            barplot.text(v + 2, i, str(v), color="black", va="center", fontsize=8)

        plt.tight_layout()
        plt.show()

    # Plot 2
    def plot_distribution_shape(self):
        plt.figure(figsize=(10, 6))
        sns.histplot(self.class_counts["Count"], bins=20, kde=True, color="purple")
        plt.title("Distribution Shape of Class Counts", fontsize=18, fontweight="bold")
        plt.xlabel("Number of Videos per Class", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.tight_layout()
        plt.show()

    # Plot 3
    def plot_box_violin(self):
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))

        sns.boxplot(y=self.class_counts["Count"], ax=ax[0], color="lightblue")
        ax[0].set_title("Boxplot: Video Counts per Class", fontsize=14, fontweight="bold")
        ax[0].set_ylabel("Number of Videos")

        sns.violinplot(y=self.class_counts["Count"], ax=ax[1], color="lightgreen")
        ax[1].set_title("Violinplot: Video Counts per Class", fontsize=14, fontweight="bold")
        ax[1].set_ylabel("")

        plt.tight_layout()
        plt.show()

    # Plot 4
    def plot_top10_pie(self):
        top10 = self.class_counts.sort_values("Count", ascending=False).head(10)

        plt.figure(figsize=(7, 7))
        plt.pie(
            top10["Count"],
            labels=top10["Class"],
            autopct="%1.1f%%",
            startangle=140,
            colors=sns.color_palette("viridis", len(top10))
        )
        plt.title("Top 10 Classes - Share of Videos", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.show()

    # Stats
    def print_outliers(self):
        Q1 = self.class_counts["Count"].quantile(0.25)
        Q3 = self.class_counts["Count"].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = self.class_counts[
            (self.class_counts["Count"] < lower_bound) |
            (self.class_counts["Count"] > upper_bound)
        ]

        if not outliers.empty:
            print("\nOutlier Classes Detected (IQR Method):")
            print(outliers.to_string(index=False))
        else:
            print("\nNo outlier classes detected.")

        return outliers

    def print_balance_score(self):
        balance_score = self.class_counts["Count"].std() / self.class_counts["Count"].mean()
        print(f"\nClass Balance Score (std/mean): {balance_score:.3f}")

        if balance_score < 0.1:
            print("Dataset is well balanced.")
        elif balance_score < 0.3:
            print("Dataset is moderately balanced.")
        else:
            print("Dataset is highly imbalanced.")

        return balance_score

    # Run all
    def run_all(self):
        self.print_info()
        self.plot_class_distribution()
        self.plot_distribution_shape()
        self.plot_box_violin()
        self.print_outliers()
        self.plot_top10_pie()
        self.print_balance_score()