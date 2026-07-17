"""
Student Marks Dataset – Complete EDA Script
By Soumya ranjan swain, Beeskilled Data Science Intern

Performs a full exploratory data analysis on the Kaggle "Student Performance" dataset.
Columns expected:
  - gender
  - race/ethnicity
  - parental level of education
  - lunch
  - test preparation course
  - math score
  - reading score
  - writing score

Workflow:
  1. Data import & quality checks
  2. Statistical analysis & metric extraction
  3. Univariate & bivariate analysis
  4. Data visualization
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress non-critical warnings for cleaner output
warnings.filterwarnings("ignore")

# ---------------------------
# Plotting configuration
# ---------------------------
sns.set(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12


def separator() -> None:
    """Print a visual separator line."""
    print("=" * 60)


def load_and_clean_data(csv_path: str | Path) -> pd.DataFrame | None:
    """
    Load the dataset, check for missing values and duplicates,
    and perform basic cleaning.

    Returns the cleaned DataFrame or None if loading fails.
    """
    csv_path = Path(csv_path)

    if not csv_path.is_file():
        print(f"\nCSV file not found: {csv_path.resolve()}")
        return None

    print("\nLoading dataset...\n")
    df = pd.read_csv(csv_path)

    separator()
    print("STUDENT MARKS DATASET – IMPORT & QUALITY CHECKS")
    separator()

    print(f"\nDataset Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")

    print("Column Names:")
    for col in df.columns:
        print(f"• {col}")

    separator()

    print("\nDataset Information (.info()):")
    df.info()

    separator()

    # Missing values
    print("\nMissing Values per Column:")
    missing = df.isna().sum()
    print(missing)
    if missing.sum() > 0:
        print("\nMissing values detected. Dropping rows with any missing values.")
        df = df.dropna()
        print(f"Rows after dropping missing values: {len(df)}")
    else:
        print("\nNo missing values found.")

    separator()

    # Duplicate rows
    num_duplicates = df.duplicated().sum()
    print(f"\nNumber of duplicate rows: {num_duplicates}")
    if num_duplicates > 0:
        print("Removing duplicate rows.")
        df = df.drop_duplicates()
        print(f"Rows after removing duplicates: {len(df)}")
    else:
        print("No duplicate rows found.")

    separator()

    print("\nNumber of unique values per column:")
    print(df.nunique())

    separator()

    return df


def statistical_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform statistical analysis:
      - Descriptive stats for scores
      - Add 'total score' and 'average score' columns
      - Compute full-marks and low-scores counts & percentages

    Returns the enriched DataFrame.
    """
    separator()
    print("STATISTICAL ANALYSIS & METRIC EXTRACTION")
    separator()

    score_cols = ["math score", "reading score", "writing score"]

    print("\nDescriptive Statistics for Score Columns:")
    print(df[score_cols].describe())

    separator()

    df["total score"] = df[score_cols].sum(axis=1)
    df["average score"] = df["total score"] / 3.0

    print("\nAdded features: 'total score' and 'average score'.")

    separator()

    print("\nStudents with Full Marks (100) in Each Subject:")
    for col in score_cols:
        count_full = (df[col] == 100).sum()
        percent_full = count_full / len(df) * 100
        print(f"{col.title()}: {count_full} students ({percent_full:.2f}%)")

    separator()

    print("\nStudents Scoring ≤ 20 in Each Subject:")
    for col in score_cols:
        count_low = (df[col] <= 20).sum()
        percent_low = count_low / len(df) * 100
        print(f"{col.title()}: {count_low} students ({percent_low:.2f}%)")

    separator()

    return df


def univariate_bivariate_analysis(df: pd.DataFrame) -> None:
    """
    Perform grouped analyses:
      - By gender
      - By lunch
      - By race/ethnicity
    """
    separator()
    print("DATA EXPLORATION – UNIVARIATE & BIVARIATE ANALYSIS")
    separator()

    score_cols = ["math score", "reading score", "writing score"]

    # By gender
    if "gender" in df.columns:
        print("\nAverage Scores by Gender:")
        gender_stats = df.groupby("gender")[score_cols].mean()
        print(gender_stats.round(2))
        print(
            "\nInterpretation: This shows how average math, reading, and writing "
            "scores differ between male and female students."
        )
    else:
        print("\n'gender' column not found; skipping gender analysis.")

    separator()

    # By lunch
    if "lunch" in df.columns:
        print("\nAverage Scores by Lunch Type:")
        lunch_stats = df.groupby("lunch")[score_cols].mean()
        print(lunch_stats.round(2))
        print(
            "\nInterpretation: Students with 'standard' lunch often have higher "
            "average scores than those with 'free/reduced' lunch, reflecting "
            "socioeconomic effects on performance."
        )
    else:
        print("\n'lunch' column not found; skipping lunch analysis.")

    separator()

    # By race/ethnicity
    if "race/ethnicity" in df.columns:
        print("\nAverage Scores by Race/Ethnicity:")
        race_stats = df.groupby("race/ethnicity")[score_cols].mean()
        print(race_stats.round(2))
        print(
            "\nInterpretation: Average scores vary across race/ethnicity groups, "
            "which may reflect systemic and environmental factors."
        )
    else:
        print("\n'race/ethnicity' column not found; skipping race/ethnicity analysis.")

    separator()


def run_visualizations(df: pd.DataFrame) -> None:
    """
    Create required visualizations:
      - 1x2: average score distribution (overall and by gender)
      - 1x3: average score by lunch, parental education, race/ethnicity
      - Violin plots for math, reading, writing
      - Bar chart: Total Average vs Math Average by gender
    """
    separator()
    print("DATA VISUALIZATION")
    separator()

    categorical_cols = [
        "gender",
        "race/ethnicity",
        "parental level of education",
        "lunch",
        "test preparation course",
    ]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # ---------------------------
    # 1x2: Average score distribution
    # ---------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # (a) Overall average score distribution
    sns.histplot(
        data=df,
        x="average score",
        kde=True,
        ax=axes[0],
        color="skyblue",
    )
    axes[0].set_title("Distribution of Average Score (Overall)")
    axes[0].set_xlabel("Average Score")
    axes[0].set_ylabel("Count")

    # (b) Average score by gender
    if "gender" in df.columns:
        sns.histplot(
            data=df,
            x="average score",
            hue="gender",
            kde=True,
            ax=axes[1],
            palette="Set2",
        )
        axes[1].set_title("Distribution of Average Score by Gender")
        axes[1].set_xlabel("Average Score")
        axes[1].set_ylabel("Count")
    else:
        axes[1].text(
            0.5,
            0.5,
            "'gender' column missing",
            transform=axes[1].transAxes,
            ha="center",
            va="center",
            fontsize=14,
        )
        axes[1].set_title("Distribution of Average Score by Gender")

    plt.tight_layout()
    plt.show()

    print(
        "\nInsight: The average score distribution is typically unimodal and roughly "
        "symmetric. When split by gender, you may observe shifts in central tendency "
        "indicating performance differences between genders."
    )

    separator()

    # ---------------------------
    # 1x3: Average score by lunch, parental education, race/ethnicity
    # ---------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # By lunch
    if "lunch" in df.columns:
        sns.boxplot(
            data=df,
            x="lunch",
            y="average score",
            ax=axes[0],
            palette="Pastel1",
        )
        axes[0].set_title("Average Score by Lunch Type")
        axes[0].set_xlabel("Lunch")
        axes[0].set_ylabel("Average Score")
    else:
        axes[0].text(
            0.5,
            0.5,
            "'lunch' column missing",
            transform=axes[0].transAxes,
            ha="center",
            va="center",
            fontsize=14,
        )
        axes[0].set_title("Average Score by Lunch Type")

    # By parental level of education
    if "parental level of education" in df.columns:
        sns.boxplot(
            data=df,
            x="parental level of education",
            y="average score",
            ax=axes[1],
            palette="Pastel2",
        )
        axes[1].set_title("Average Score by Parental Education")
        axes[1].set_xlabel("Parental Level of Education")
        axes[1].set_ylabel("Average Score")
        plt.setp(
            axes[1].xaxis.get_majorticklabels(),
            rotation=45,
            ha="right",
        )
    else:
        axes[1].text(
            0.5,
            0.5,
            "'parental level of education' missing",
            transform=axes[1].transAxes,
            ha="center",
            va="center",
            fontsize=14,
        )
        axes[1].set_title("Average Score by Parental Education")

    # By race/ethnicity
    if "race/ethnicity" in df.columns:
        sns.boxplot(
            data=df,
            x="race/ethnicity",
            y="average score",
            ax=axes[2],
            palette="Set3",
        )
        axes[2].set_title("Average Score by Race/Ethnicity")
        axes[2].set_xlabel("Race/Ethnicity")
        axes[2].set_ylabel("Average Score")
        plt.setp(
            axes[2].xaxis.get_majorticklabels(),
            rotation=45,
            ha="right",
        )
    else:
        axes[2].text(
            0.5,
            0.5,
            "'race/ethnicity' column missing",
            transform=axes[2].transAxes,
            ha="center",
            va="center",
            fontsize=14,
        )
        axes[2].set_title("Average Score by Race/Ethnicity")

    plt.tight_layout()
    plt.show()

    print(
        "\nInsight: Boxplots reveal how average scores vary across socioeconomic "
        "(lunch), educational background (parental education), and demographic "
        "(race/ethnicity) segments. Median lines and spread indicate both central "
        "tendency and variability."
    )

    separator()

    # ---------------------------
    # Bar chart: Total Average vs Math Average by Gender
    # ---------------------------
    if "gender" in df.columns:
        gender_agg = df.groupby("gender").agg(
            total_avg=("average score", "mean"),
            math_avg=("math score", "mean"),
        ).reset_index()

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        bar_width = 0.35
        x = np.arange(len(gender_agg))

        ax.bar(
            x - bar_width / 2,
            gender_agg["total_avg"],
            width=bar_width,
            label="Total Average",
            color="steelblue",
        )
        ax.bar(
            x + bar_width / 2,
            gender_agg["math_avg"],
            width=bar_width,
            label="Math Average",
            color="coral",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(gender_agg["gender"])
        ax.set_xlabel("Gender")
        ax.set_ylabel("Average Score")
        ax.set_title("Total Average vs Math Average by Gender")
        ax.legend()

        plt.tight_layout()
        plt.show()

        print(
            "\nInsight: This bar chart compares overall performance (total average) "
            "against math-specific performance by gender. Differences highlight whether "
            "gaps are driven primarily by math or are consistent across all subjects."
        )
    else:
        print("\n'gender' column missing; skipping Total vs Math Average by Gender plot.")

    separator()


def run_eda(csv_path: str | Path) -> None:
    """
    Main EDA pipeline:
      1. Load & clean data
      2. Statistical analysis
      3. Univariate & bivariate analysis
      4. Visualizations
    """
    df = load_and_clean_data(csv_path)
    if df is None:
        return

    df = statistical_analysis(df)
    univariate_bivariate_analysis(df)
    run_visualizations(df)

    print("\nEDA completed successfully.")


if __name__ == "__main__":
    path = r"Week1_Project\StudentsPerformance.csv"
    # user_path = input("Enter CSV file path (press Enter for default): ").strip()

    run_eda(path)