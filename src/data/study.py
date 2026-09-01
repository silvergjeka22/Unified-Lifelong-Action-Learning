import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.config.config import DATASET_ROOT


class DatasetStudy:
    """
    Summarise the UCF101 CSV split files: class distribution, balance, outliers.

    Usage:
        study = DatasetStudy()
        study.run_all()
    """

    def __init__(self, dataset_root=DATASET_ROOT):
        self.dataset_root = dataset_root
        self.df = None
        self.class_counts = None
        self._load()

    def _load(self):
        frames = []
        for split in ["train", "val", "test"]:
            path = os.path.join(self.dataset_root, f"{split}.csv")
            if os.path.exists(path):
                frames.append(pd.read_csv(path))
        self.df = pd.concat(frames, ignore_index=True)
        counts = self.df["label"].value_counts().reset_index()
        counts.columns = ["Class", "Count"]
        self.class_counts = counts.sort_values("Count").reset_index(drop=True)
        print(f"Total videos  : {len(self.df)}")
        print(f"Total classes : {self.class_counts['Class'].nunique()}")

    def plot_class_distribution(self):
        plt.figure(figsize=(12, max(10, len(self.class_counts) * 0.22)))
        sns.barplot(data=self.class_counts, y="Class", x="Count",
                    hue="Class", palette="viridis", legend=False)
        plt.title("UCF101 class distribution (sorted)", fontweight="bold")
        plt.xlabel("videos"); plt.ylabel("class")
        plt.tight_layout(); plt.show()

    def plot_distribution_shape(self):
        plt.figure(figsize=(10, 5))
        sns.histplot(self.class_counts["Count"], bins=20, kde=True, color="purple")
        plt.title("Distribution of per-class video counts", fontweight="bold")
        plt.xlabel("videos per class"); plt.ylabel("frequency")
        plt.tight_layout(); plt.show()

    def plot_box_violin(self):
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        sns.boxplot(y=self.class_counts["Count"], ax=ax[0], color="lightblue")
        ax[0].set_title("Boxplot: videos per class")
        sns.violinplot(y=self.class_counts["Count"], ax=ax[1], color="lightgreen")
        ax[1].set_title("Violinplot: videos per class")
        plt.tight_layout(); plt.show()

    def print_balance_score(self):
        score = self.class_counts["Count"].std() / self.class_counts["Count"].mean()
        print(f"Balance score (std/mean): {score:.3f}")
        return score

    def run_all(self):
        self.plot_class_distribution()
        self.plot_distribution_shape()
        self.plot_box_violin()
        self.print_balance_score()
