# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np

# Set page configuration
st.set_page_config(page_title="Netflix AI Dashboard", layout="wide")

# Load cleaned data
@st.cache_data
def load_data():
    df = pd.read_csv("netflix_cleaned.csv")
    df['content'] = df['title'] + ' ' + df['listed_in']
    return df

df = load_data()

# TF-IDF for recommender and clustering
@st.cache_data
def get_tfidf_matrix(content_series):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(content_series)
    return tfidf_matrix

tfidf_matrix = get_tfidf_matrix(df['content'])

# ----------------------------------------
# Recommender Function
# ----------------------------------------
def recommend_titles(title, df, tfidf_matrix, top_n=5):
    title = title.lower()
    idx = df[df['title'].str.lower() == title].index
    if len(idx) == 0:
        return None
    idx = idx[0]
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_indices = cosine_sim.argsort()[-top_n-1:-1][::-1]
    recommended = df.iloc[similar_indices][['title', 'type', 'listed_in', 'country']]
    return recommended

# ----------------------------------------
# Forecasting Function
# ----------------------------------------
def run_forecasting_model():
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
    df['year_added'] = df['date_added'].dt.year
    yearly_trend = df.groupby('year_added').size().reset_index(name='content_count')
    X = yearly_trend['year_added'].values.reshape(-1, 1)
    y = yearly_trend['content_count'].values
    model = LinearRegression()
    model.fit(X, y)
    future_years = np.arange(2025, 2031).reshape(-1, 1)
    future_preds = model.predict(future_years)
    future_df = pd.DataFrame({'year_added': future_years.flatten(), 'content_count': future_preds.astype(int)})
    combined = pd.concat([yearly_trend, future_df])

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=combined, x='year_added', y='content_count', ax=ax, marker='o', label='Forecast')
    ax.axvline(x=2024.5, linestyle='--', color='gray', label='Prediction Starts')
    ax.set_title("Forecast of Netflix Content Additions")
    ax.set_xlabel("Year")
    ax.set_ylabel("Titles Added")
    ax.legend()
    ax.grid(True)

    # Summary
    latest_year = int(yearly_trend['year_added'].max())
    latest_content = int(yearly_trend[yearly_trend['year_added'] == latest_year]['content_count'])
    growth_rate = (model.coef_[0] / latest_content) * 100
    summary = f"""
📈 Netflix added {latest_content} titles in {latest_year}.
📊 Expected to reach {int(future_preds[-1])} titles in 2030.
📈 Estimated growth rate: {growth_rate:.2f}% annually.
"""
    return fig, summary

# ----------------------------------------
# Clustering Function
# ----------------------------------------
@st.cache_data
def get_clusters(df, _tfidf_matrix, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(_tfidf_matrix)
    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(_tfidf_matrix.toarray())
    df['pca_x'] = reduced[:, 0]
    df['pca_y'] = reduced[:, 1]
    return df

df = get_clusters(df, tfidf_matrix)

# ----------------------------------------
# Streamlit UI
# ----------------------------------------

st.title("🎬 Netflix AI-Powered Dashboard")

# Sidebar
menu = st.sidebar.radio("Navigate", ["Home", "EDA", "Recommender", "Clustering", "Forecasting"])

# ----------------------------------------
# Home
# ----------------------------------------
if menu == "Home":
    st.subheader("📺 Explore Netflix like never before!")
    st.markdown("""
Welcome to the **Netflix AI Dashboard** — your one-stop app to explore, analyze, and predict trends in Netflix's vast content library using **Machine Learning and NLP**.
    
🔎 Analyze genres, trends, and ratings  
🤖 Get smart title recommendations  
📈 Forecast future content growth  
🧠 Visualize AI-based clusters  
    """)

# ----------------------------------------
# EDA
# ----------------------------------------
elif menu == "EDA":
    st.subheader("📊 Exploratory Data Analysis")

    type_counts = df['type'].value_counts()
    st.bar_chart(type_counts)

    top_countries = df['country'].value_counts().head(10)
    st.bar_chart(top_countries)

    rating_counts = df['rating'].value_counts()
    st.bar_chart(rating_counts)

    top_genres = pd.Series(','.join(df['listed_in']).split(',')).str.strip().value_counts().head(10)
    st.bar_chart(top_genres)

# ----------------------------------------
# Recommender
# ----------------------------------------
elif menu == "Recommender":
    st.subheader("🎯 Content-Based Recommender")

    user_input = st.text_input("Enter a Netflix Title:")
    if user_input:
        results = recommend_titles(user_input, df, tfidf_matrix)
        if results is not None:
            st.write(f"Top recommendations for **{user_input.title()}**:")
            st.dataframe(results)
        else:
            st.warning("No match found. Please try another title.")

# ----------------------------------------
# Clustering
# ----------------------------------------
elif menu == "Clustering":
    st.subheader("🧠 Netflix Content Clustering (K-Means)")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x='pca_x', y='pca_y', hue='cluster', palette='Set2', ax=ax)
    ax.set_title("Clustering of Netflix Titles")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.legend(title="Cluster")
    st.pyplot(fig)

# ----------------------------------------
# Forecasting
# ----------------------------------------
elif menu == "Forecasting":
    st.subheader("📈 Forecasting Future Netflix Growth")
    fig, summary = run_forecasting_model()
    st.pyplot(fig)
    st.markdown("**AI Summary:**")
    st.success(summary)
