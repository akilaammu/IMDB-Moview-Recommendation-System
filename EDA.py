import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


# ==========================================================
# 1. LOAD DATASET
# ==========================================================

df = pd.read_csv(
    "C:/Users/Thang/Desktop/MiniProject5/Dataset/imdb_movies_2024.csv"
)


# ==========================================================
# 2. DATASET OVERVIEW
# ==========================================================

print("=" * 60)
print("IMDb 2024 MOVIE DATASET - EDA")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nNumber of Movies:", len(df))

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Records:")
print(df.head())


# ==========================================================
# 3. DATA TYPES
# ==========================================================

print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)

print(df.dtypes)


# ==========================================================
# 4. DATASET INFORMATION
# ==========================================================

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

df.info()


# ==========================================================
# 5. MISSING VALUE ANALYSIS
# ==========================================================

print("\n" + "=" * 60)
print("MISSING VALUE ANALYSIS")
print("=" * 60)

missing_values = df.isnull().sum()

print(missing_values)

print("\nTotal Missing Values:")
print(missing_values.sum())


# ==========================================================
# 6. DUPLICATE ANALYSIS
# ==========================================================

print("\n" + "=" * 60)
print("DUPLICATE ANALYSIS")
print("=" * 60)

print(
    "Duplicate Rows:",
    df.duplicated().sum()
)

print(
    "Duplicate Movie Names:",
    df["Movie Name"].duplicated().sum()
)

print(
    "Duplicate Storylines:",
    df["Storyline"].duplicated().sum()
)


# ==========================================================
# 7. EMPTY STORYLINE ANALYSIS
# ==========================================================

empty_storylines = (
    df["Storyline"]
    .fillna("")
    .str.strip()
    .eq("")
    .sum()
)

empty_cleaned_storylines = (
    df["Cleaned_Storyline"]
    .fillna("")
    .str.strip()
    .eq("")
    .sum()
)

print("\n" + "=" * 60)
print("EMPTY STORYLINE ANALYSIS")
print("=" * 60)

print(
    "Empty Storylines:",
    empty_storylines
)

print(
    "Empty Cleaned Storylines:",
    empty_cleaned_storylines
)


# ==========================================================
# 8. STORYLINE CHARACTER LENGTH
# ==========================================================

df["Storyline_Length"] = (
    df["Storyline"]
    .fillna("")
    .str.len()
)


# ==========================================================
# 9. STORYLINE WORD COUNT
# ==========================================================

df["Storyline_Word_Count"] = (
    df["Storyline"]
    .fillna("")
    .str.split()
    .str.len()
)


# ==========================================================
# 10. CLEANED STORYLINE WORD COUNT
# ==========================================================

df["Cleaned_Word_Count"] = (
    df["Cleaned_Storyline"]
    .fillna("")
    .str.split()
    .str.len()
)


# ==========================================================
# 11. STORYLINE STATISTICS
# ==========================================================

print("\n" + "=" * 60)
print("STORYLINE STATISTICS")
print("=" * 60)

print(
    df[
        [
            "Storyline_Length",
            "Storyline_Word_Count",
            "Cleaned_Word_Count"
        ]
    ].describe()
)


# ==========================================================
# 12. AVERAGE STORYLINE INFORMATION
# ==========================================================

print("\n" + "=" * 60)
print("AVERAGE STORYLINE INFORMATION")
print("=" * 60)

print(
    "Average Storyline Characters:",
    round(df["Storyline_Length"].mean(), 2)
)

print(
    "Average Storyline Words:",
    round(df["Storyline_Word_Count"].mean(), 2)
)

print(
    "Average Cleaned Storyline Words:",
    round(df["Cleaned_Word_Count"].mean(), 2)
)


# ==========================================================
# 13. SHORTEST AND LONGEST STORYLINES
# ==========================================================

print("\n" + "=" * 60)
print("SHORTEST STORYLINES")
print("=" * 60)

shortest = df.nsmallest(
    10,
    "Storyline_Word_Count"
)

print(
    shortest[
        [
            "Movie Name",
            "Storyline_Word_Count"
        ]
    ]
)


print("\n" + "=" * 60)
print("LONGEST STORYLINES")
print("=" * 60)

longest = df.nlargest(
    10,
    "Storyline_Word_Count"
)

print(
    longest[
        [
            "Movie Name",
            "Storyline_Word_Count"
        ]
    ]
)


# ==========================================================
# 14. TOP FREQUENT WORDS
# ==========================================================

all_words = " ".join(
    df["Cleaned_Storyline"]
    .fillna("")
).split()

word_counts = Counter(all_words)

top_words = word_counts.most_common(20)

print("\n" + "=" * 60)
print("TOP 20 MOST FREQUENT WORDS")
print("=" * 60)

for word, count in top_words:
    print(
        f"{word:<20} : {count}"
    )


# ==========================================================
# 15. VISUALIZATION
# Storyline Word Count Distribution
# ==========================================================

plt.figure(figsize=(8, 5))

plt.hist(
    df["Storyline_Word_Count"],
    bins=30
)

plt.title(
    "Distribution of Movie Storyline Word Count"
)

plt.xlabel(
    "Number of Words"
)

plt.ylabel(
    "Number of Movies"
)

plt.tight_layout()

plt.show()


# ==========================================================
# 16. VISUALIZATION
# Cleaned vs Original Word Count
# ==========================================================

plt.figure(figsize=(8, 5))

plt.hist(
    df["Storyline_Word_Count"],
    bins=30,
    alpha=0.6,
    label="Original Storyline"
)

plt.hist(
    df["Cleaned_Word_Count"],
    bins=30,
    alpha=0.6,
    label="Cleaned Storyline"
)

plt.title(
    "Original vs Cleaned Storyline Word Count"
)

plt.xlabel(
    "Number of Words"
)

plt.ylabel(
    "Number of Movies"
)

plt.legend()

plt.tight_layout()

plt.show()


# ==========================================================
# 17. VISUALIZATION
# Top 20 Frequent Words
# ==========================================================

words = [
    item[0]
    for item in top_words
]

counts = [
    item[1]
    for item in top_words
]


plt.figure(figsize=(10, 6))

plt.barh(
    words[::-1],
    counts[::-1]
)

plt.title(
    "Top 20 Most Frequent Words in Movie Storylines"
)

plt.xlabel(
    "Frequency"
)

plt.ylabel(
    "Word"
)

plt.tight_layout()

plt.show()


# ==========================================================
# 18. DATASET SUMMARY
# ==========================================================

print("\n" + "=" * 60)
print("FINAL DATASET SUMMARY")
print("=" * 60)

print(
    "Total Movies:",
    len(df)
)

print(
    "Total Columns:",
    len(df.columns)
)

print(
    "Missing Storylines:",
    df["Storyline"].isnull().sum()
)

print(
    "Duplicate Rows:",
    df.duplicated().sum()
)

print(
    "Duplicate Movie Names:",
    df["Movie Name"].duplicated().sum()
)

print(
    "Average Storyline Words:",
    round(
        df["Storyline_Word_Count"].mean(),
        2
    )
)

print(
    "Average Cleaned Storyline Words:",
    round(
        df["Cleaned_Word_Count"].mean(),
        2
    )
)

print("\nEDA Completed Successfully!")
