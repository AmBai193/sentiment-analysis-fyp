import os
import ast
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure plots render with a clean style
sns.set(style="whitegrid")

# ----------------------------
# Configuration:
# List your CSV files here, and map each to its sentiment column
# ----------------------------
datasets = {
    "combined_sentiment_nb_textblob.csv":        "Sentiment",
    "combined_sentiment_nb_vader.csv":           "Sentiment",
    "combined_textblob.csv":                     "predicted_label",
    "combined_vader.csv":                        "predicted_label",
    "bert_predictions_all(TextBlob).csv":        "BERT_Predicted_Sentiment",
    "bert_predictions_all(VADER).csv":           "BERT_Predicted_Sentiment",
    "bert_predictions_all(ChatGpt).csv":         "BERT_Predicted_Sentiment"
}



# ----------------------------
# Helper Functions
# ----------------------------

def load_and_prepare(df_path, sentiment_col):
    """
    1. Load CSV from df_path
    2. Parse 'Mentioned_Parties' (string-encoded list) into actual Python lists.
    3. Explode each row so that each (party, sentiment) pair is its own row.
    4. Normalize 'Sentiment' values to exactly 'Positive', 'Neutral', or 'Negative'.
    Returns a DataFrame with columns ['Party', 'Sentiment'].
    """
    df = pd.read_csv(df_path)

    # Parse Mentioned_Parties from string to list
    df["Mentioned_Parties"] = df["Mentioned_Parties"].apply(ast.literal_eval)

    # Explode parties
    exploded = df.explode("Mentioned_Parties").rename(columns={"Mentioned_Parties": "Party"})

    # Keep only needed columns
    exploded = exploded[["Party", sentiment_col]].copy()
    exploded = exploded.rename(columns={sentiment_col: "Sentiment"})

    # Normalize sentiment: strip whitespace, lowercase, then capitalize
    exploded["Sentiment"] = (
        exploded["Sentiment"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "positive": "Positive",
            "neutral": "Neutral",
            "negative": "Negative"
        })
    )

    # Drop any rows where normalization failed (i.e., got NaN)
    exploded = exploded.dropna(subset=["Sentiment"])

    # Drop rows where Party is empty or 'OTHER'
    exploded = exploded[
        (exploded["Party"].astype(str).str.strip() != "") &
        (exploded["Party"] != "OTHER")
        ]

    return exploded


def party_sentiment_counts(df_exploded):
    """
    Given an exploded DataFrame with columns ['Party', 'Sentiment'],
    returns a pivot dataframe: index=Party, columns=[Positive, Neutral, Negative], values=count
    """
    pivot = (
        df_exploded
        .groupby(["Party", "Sentiment"])
        .size()
        .unstack(fill_value=0)[["Positive", "Neutral", "Negative"]]
        .sort_index()
    )
    return pivot


def plot_stacked_bar(pivot_df, title=None):
    """
    Plot a stacked bar chart of sentiment counts by party.
    pivot_df: rows=Party, columns=['Positive','Neutral','Negative']
    """
    ax = pivot_df.plot(
        kind="bar",
        stacked=True,
        figsize=(12, 6),
        colormap="coolwarm"
    )
    ax.set_xlabel("Party")
    ax.set_ylabel("Number of Comments")
    if title:
        ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_pie_charts(pivot_df, top_n=5, title_suffix=""):
    """
    For the top_n parties by total comment count, plot individual pie charts of sentiment distribution.
    """
    totals = pivot_df.sum(axis=1).sort_values(ascending=False)
    top_parties = totals.head(top_n).index.tolist()
    for party in top_parties:
        counts = pivot_df.loc[party]
        plt.figure(figsize=(4, 4))
        counts.plot.pie(
            labels=[f"{sent} ({counts[sent]})" for sent in counts.index],
            autopct="%1.1f%%",
            startangle=140,
            colors=["#2ca02c", "#1f77b4", "#d62728"]
        )
        plt.title(f"{party} Sentiment Breakdown {title_suffix}")
        plt.ylabel("")  # Hide default y-label
        plt.tight_layout()
        plt.show()


def most_frequent_parties(df_exploded, top_n=10):
    """
    Returns a Series of top_n parties by mention count.
    """
    counts = df_exploded["Party"].value_counts()
    return counts.head(top_n)


def plot_top_parties(counts_series, title=None):
    """
    Plot a bar chart of top-mentioned parties.
    """
    plt.figure(figsize=(10, 5))
    sns.barplot(
        x=counts_series.values,
        y=counts_series.index,
        palette="viridis"
    )
    plt.xlabel("Number of Mentions")
    plt.ylabel("Party")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


def sentiment_ratio(pivot_df):
    """
    Given a pivot_df with columns ['Positive','Neutral','Negative'] per party, compute ratio:
    ratio = Positive / (Positive+Negative). Returns a DataFrame with additional columns:
      - 'Total'
      - 'Positive_Count','Neutral_Count','Negative_Count'
      - 'Positive_Ratio'  (Positive / Total)
      - 'Negative_Ratio'  (Negative / Total)
      - 'Neutral_Ratio'   (Neutral / Total)
      - 'Pos_vs_Neg_Ratio' (Positive / Negative)  (handle division by zero)
    """
    df = pivot_df.copy()
    df["Total"] = df.sum(axis=1)
    df["Positive_Count"] = df["Positive"]
    df["Neutral_Count"] = df["Neutral"]
    df["Negative_Count"] = df["Negative"]
    df["Positive_Ratio"] = df["Positive"] / df["Total"]
    df["Neutral_Ratio"] = df["Neutral"] / df["Total"]
    df["Negative_Ratio"] = df["Negative"] / df["Total"]
    df["Pos_vs_Neg_Ratio"] = df.apply(
        lambda row: row["Positive"] / row["Negative"] if row["Negative"] > 0 else float("inf"),
        axis=1
    )
    return df.sort_values("Positive_Ratio", ascending=False)


# ----------------------------
# Main Analysis Loop
# ----------------------------
for filename, sentiment_col in datasets.items():
    if not os.path.exists(filename):
        print(f"File '{filename}' not found. Skipping.")
        continue

    print(f"\n--- Analyzing {filename} (sentiment column: '{sentiment_col}') ---")
    exploded_df = load_and_prepare(filename, sentiment_col)

    # 1. Party-wise Sentiment Distribution
    pivot = party_sentiment_counts(exploded_df)
    print("\nParty-wise sentiment counts:\n", pivot)

    # Plot stacked bar chart
    plot_stacked_bar(pivot, title=f"Sentiment Distribution by Party\n({filename})")

    # Optionally plot top 5 parties pie charts
    plot_pie_charts(pivot, top_n=5, title_suffix=f"({filename})")

    # 2. Most Frequently Mentioned Party
    top_parties = most_frequent_parties(exploded_df, top_n=10)
    print("\nTop 10 most frequently mentioned parties:\n", top_parties)
    plot_top_parties(top_parties, title=f"Top 10 Mentioned Parties\n({filename})")

    # 3. Party Comparison by Sentiment Ratio
    ratio_df = sentiment_ratio(pivot)
    print("\nParty sentiment ratios:\n", ratio_df[[
        "Positive_Count", "Neutral_Count", "Negative_Count",
        "Positive_Ratio", "Neutral_Ratio", "Negative_Ratio", "Pos_vs_Neg_Ratio"
    ]])

    # Identify most favorable (highest Positive_Ratio) and most critical (lowest Positive_Ratio)
    most_favorable = ratio_df.head(1)
    most_critical = ratio_df.tail(1)
    print("\nMost Favorable Party:\n", most_favorable[["Positive_Ratio"]])
    print("Most Critical Party:\n", most_critical[["Positive_Ratio"]])

    # Barplot of Positive_Ratio for all parties
    plt.figure(figsize=(12, 6))
    sns.barplot(
        x=ratio_df["Positive_Ratio"],
        y=ratio_df.index,
        palette="summer"
    )
    plt.xlabel("Positive Ratio (Positive / Total)")
    plt.ylabel("Party")
    plt.title(f"Positive Ratio by Party\n({filename})")
    plt.tight_layout()
    plt.show()
