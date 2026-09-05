import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================================
# 1. LOAD DATASET
# ==========================================================

df = pd.read_csv(
    "C:/Users/Thang/Desktop/MiniProject5/Dataset/imdb_movies_2024.csv"
)

print("=" * 60)
print("IMDb MOVIE RECOMMENDATION SYSTEM")
print("=" * 60)

print("\nDataset Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ==========================================================
# 2. PREPARE STORYLINE DATA
# ==========================================================

storylines = (
    df["Cleaned_Storyline"]
    .fillna("")
    .astype(str)
)


# ==========================================================
# 3. TF-IDF VECTORIZATION
# ==========================================================

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

tfidf_matrix = tfidf.fit_transform(
    storylines
)

print(
    "\nTF-IDF Matrix Shape:",
    tfidf_matrix.shape
)


# ==========================================================
# 4. SAVE TF-IDF VECTORIZER
# ==========================================================

with open(
    "C:/Users/Thang/Desktop/MiniProject5/models/tfidf_vectorizer.pkl",
    "wb"
) as file:

    pickle.dump(
        tfidf,
        file
    )


# ==========================================================
# 5. SAVE TF-IDF MATRIX
# ==========================================================

with open(
    "C:/Users/Thang/Desktop/MiniProject5/models/tfidf_matrix.pkl",
    "wb"
) as file:

    pickle.dump(
        tfidf_matrix,
        file
    )


print(
    "\nTF-IDF vectorizer saved successfully."
)

print(
    "TF-IDF matrix saved successfully."
)


# ==========================================================
# 6. MOVIE-TO-MOVIE RECOMMENDATION
# ==========================================================

def recommend_movies(
    movie_index,
    top_n=5
):
    """
    Recommend movies similar to an existing movie
    in the dataset.

    Parameters:
        movie_index : int
            Index of the movie in the dataset.

        top_n : int
            Number of recommendations.

    Returns:
        DataFrame containing recommended movies
        and similarity scores.
    """

    # Calculate similarity between selected movie
    # and all movies

    similarity_scores = cosine_similarity(
        tfidf_matrix[movie_index],
        tfidf_matrix
    )[0]


    # Sort similarity scores from highest to lowest

    similar_indices = (
        similarity_scores
        .argsort()[::-1]
    )


    # Remove the selected movie itself

    similar_indices = [
        i
        for i in similar_indices
        if i != movie_index
    ]


    # Select top N movies

    top_indices = (
        similar_indices[:top_n]
    )


    # Create recommendation dataframe

    recommendations = df.iloc[
        top_indices
    ][
        [
            "Movie Name",
            "Storyline"
        ]
    ].copy()


    # Add similarity score

    recommendations["Similarity"] = [
        similarity_scores[i]
        for i in top_indices
    ]


    return recommendations


# ==========================================================
# 7. STORYLINE-BASED RECOMMENDATION
# ==========================================================

def recommend_by_storyline(
    cleaned_storyline,
    top_n=5
):
    """
    Recommend movies based on a user-provided
    cleaned storyline.

    The storyline is supplied dynamically by the
    Streamlit application.

    Parameters:
        cleaned_storyline : str
            Preprocessed user storyline.

        top_n : int
            Number of recommendations.

    Returns:
        DataFrame containing recommended movies
        and similarity scores.
    """

    # Convert user storyline into TF-IDF vector

    user_vector = tfidf.transform(
        [cleaned_storyline]
    )


    # Calculate similarity with all movies

    similarity_scores = cosine_similarity(
        user_vector,
        tfidf_matrix
    )[0]


    # Rank movies

    top_indices = (
        similarity_scores
        .argsort()[::-1][:top_n]
    )


    # Create recommendation dataframe

    recommendations = df.iloc[
        top_indices
    ][
        [
            "Movie Name",
            "Storyline"
        ]
    ].copy()


    # Add similarity score

    recommendations["Similarity"] = [
        similarity_scores[i]
        for i in top_indices
    ]


    return recommendations


# ==========================================================
# 8. MODEL INFORMATION
# ==========================================================

print("\n" + "=" * 60)
print("MODEL INFORMATION")
print("=" * 60)

print(
    "Vectorizer: TF-IDF"
)

print(
    "Maximum Features:",
    5000
)

print(
    "N-gram Range:",
    "(1, 2)"
)

print(
    "Number of TF-IDF Features:",
    len(tfidf.get_feature_names_out())
)

print(
    "Number of Movies:",
    len(df)
)

print(
    "\nRecommendation model is ready."
)
