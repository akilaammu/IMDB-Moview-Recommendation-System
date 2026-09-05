import streamlit as st
import pandas as pd
import pickle

from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import clean_text


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="IMDb Movie Recommendation",
    page_icon="🎬",
    layout="wide"
)


# ==========================================================
# LOAD DATASET
# ==========================================================

@st.cache_data
def load_dataset():

    return pd.read_csv(
        "C:/Users/Thang/Desktop/MiniProject5/Dataset/imdb_movies_2024.csv"
    )


df = load_dataset()


# ==========================================================
# LOAD TF-IDF VECTORIZER
# ==========================================================

@st.cache_resource
def load_tfidf_vectorizer():

    with open(
        "C:/Users/Thang/Desktop/MiniProject5/models/tfidf_vectorizer.pkl",
        "rb"
    ) as file:

        return pickle.load(file)


tfidf = load_tfidf_vectorizer()


# ==========================================================
# LOAD TF-IDF MATRIX
# ==========================================================

@st.cache_resource
def load_tfidf_matrix():

    with open(
        "C:/Users/Thang/Desktop/MiniProject5/models/tfidf_matrix.pkl",
        "rb"
    ) as file:

        return pickle.load(file)


tfidf_matrix = load_tfidf_matrix()


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("🎬 IMDb Recommendation")

st.sidebar.markdown(
    """
    ### Navigation

    **🎯 Movie Recommendation**

    **📊 Dataset Insights**

    **ℹ️ About Project**
    """
)

st.sidebar.divider()

st.sidebar.markdown(
    """
    ### Project Information

    **Domain:** Entertainment / Data Analytics

    **Technique:** TF-IDF + Cosine Similarity

    **Dataset:** IMDb 2024 Movies

    **Recommendations:** Top 5
    """
)


# ==========================================================
# MAIN TITLE
# ==========================================================

st.title(
    "🎬 IMDb Movie Recommendation System"
)

st.write(
    "Enter a movie storyline and discover the "
    "5 most similar movies based on storyline similarity."
)


# ==========================================================
# KPI CARDS
# ==========================================================

total_movies = len(df)

total_storylines = (
    df["Storyline"]
    .notna()
    .sum()
)

missing_storylines = (
    df["Storyline"]
    .isna()
    .sum()
)

avg_storyline_words = (
    df["Storyline"]
    .fillna("")
    .str.split()
    .str.len()
    .mean()
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "🎬 Total Movies",
        f"{total_movies:,}"
    )


with col2:

    st.metric(
        "📝 Storylines",
        f"{total_storylines:,}"
    )


with col3:

    st.metric(
        "⚠️ Missing Storylines",
        f"{missing_storylines:,}"
    )


with col4:

    st.metric(
        "📖 Avg. Storyline Words",
        f"{avg_storyline_words:.0f}"
    )


st.divider()


# ==========================================================
# STORYLINE INPUT
# ==========================================================

st.subheader(
    "🔍 Enter Movie Storyline"
)

storyline = st.text_area(
    "Movie Storyline",
    height=150,
    placeholder=(
        "Enter a movie storyline here..."
    )
)


# ==========================================================
# RECOMMENDATION BUTTON
# ==========================================================

if st.button(
    "🔍 Recommend Movies",
    use_container_width=True
):

    # ------------------------------------------------------
    # CHECK EMPTY INPUT
    # ------------------------------------------------------

    if storyline.strip() == "":

        st.warning(
            "Please enter a movie storyline."
        )


    else:

        # --------------------------------------------------
        # CLEAN USER INPUT
        # --------------------------------------------------

        cleaned_input = clean_text(
            storyline
        )


        # --------------------------------------------------
        # CHECK CLEANED INPUT
        # --------------------------------------------------

        if cleaned_input.strip() == "":

            st.warning(
                "Please enter a meaningful movie storyline."
            )


        else:

            # ----------------------------------------------
            # CONVERT INPUT TO TF-IDF VECTOR
            # ----------------------------------------------

            user_vector = tfidf.transform(
                [cleaned_input]
            )


            # ----------------------------------------------
            # CALCULATE COSINE SIMILARITY
            # ----------------------------------------------

            similarity_scores = cosine_similarity(
                user_vector,
                tfidf_matrix
            )[0]


            # ----------------------------------------------
            # GET TOP 5 MOVIES
            # ----------------------------------------------

            top_indices = (
                similarity_scores
                .argsort()[::-1][:5]
            )


            # ----------------------------------------------
            # DISPLAY RESULTS
            # ----------------------------------------------

            st.subheader(
                "🎬 Top 5 Recommended Movies"
            )


            for rank, index in enumerate(
                top_indices,
                start=1
            ):

                movie_name = df.iloc[index][
                    "Movie Name"
                ]

                movie_storyline = df.iloc[index][
                    "Storyline"
                ]

                similarity = (
                    similarity_scores[index] * 100
                )


                # ------------------------------------------
                # MOVIE TITLE
                # ------------------------------------------

                st.markdown(
                    f"### {rank}. {movie_name}"
                )


                # ------------------------------------------
                # STORYLINE
                # ------------------------------------------

                st.write(
                    movie_storyline
                )


                # ------------------------------------------
                # SIMILARITY SCORE
                # ------------------------------------------

                st.write(
                    f"**Similarity Score:** "
                    f"{similarity:.2f}%"
                )


                # ------------------------------------------
                # PROGRESS BAR
                # ------------------------------------------

                st.progress(
                    min(
                        int(similarity),
                        100
                    )
                )


                st.divider()


# ==========================================================
# DATASET INSIGHTS
# ==========================================================

with st.expander(
    "📊 View Dataset Insights"
):

    st.subheader(
        "Dataset Overview"
    )

    col1, col2 = st.columns(2)


    with col1:

        st.write(
            "**Dataset Shape:**"
        )

        st.write(
            f"{df.shape[0]:,} rows × "
            f"{df.shape[1]:,} columns"
        )


    with col2:

        st.write(
            "**TF-IDF Features:**"
        )

        st.write(
            f"{len(tfidf.get_feature_names_out()):,}"
        )


    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(
        df[
            [
                "Movie Name",
                "Storyline"
            ]
        ].head(10),
        use_container_width=True
    )


# ==========================================================
# ABOUT PROJECT
# ==========================================================

with st.expander(
    "ℹ️ About This Project"
):

    st.markdown(
        """
        ### IMDb Movie Recommendation System

        This project recommends movies based on the
        similarity between movie storylines.

        ### Methodology

        1. IMDb 2024 movie data was collected.
        2. Storylines were cleaned using NLP preprocessing.
        3. TF-IDF was used to convert storylines into
           numerical vectors.
        4. Cosine Similarity was used to measure similarity.
        5. The top 5 most similar movies are recommended.

        ### Technologies

        - Python
        - Pandas
        - Scikit-learn
        - NLP
        - TF-IDF
        - Cosine Similarity
        - Streamlit
        - Selenium (optional data collection)
        """
    )
