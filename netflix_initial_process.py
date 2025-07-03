import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import datetime as dt

# For AI/NLP later
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("netflix1.csv")

print(df.head())
print(df.shape)
print(df.dtypes)

df.drop_duplicates(inplace=True)

# Convert date_added to datetime
df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')

# Fill or drop NaNs in important columns
df.dropna(subset=['title', 'country', 'rating', 'listed_in'], inplace=True)
df['director'] = df['director'].fillna("Unknown")

# Reset index after drops
df.reset_index(drop=True, inplace=True)

# Extract Year, Month, Day from date_added
df['year_added'] = df['date_added'].dt.year
df['month_added'] = df['date_added'].dt.month
df['day_added'] = df['date_added'].dt.day

# Calculate content age
df['content_age'] = 2025 - df['release_year']

# Number of genres
df['num_genres'] = df['listed_in'].apply(lambda x: len(x.split(',')))

# Clean up whitespace
df['listed_in'] = df['listed_in'].apply(lambda x: ','.join([g.strip() for g in x.split(',')]))

# Save cleaned dataset
df.to_csv("netflix_cleaned.csv", index=False)

import seaborn as sns
import matplotlib.pyplot as plt

type_counts = df['type'].value_counts()

plt.figure(figsize=(6, 4))
sns.barplot(x=type_counts.index, y=type_counts.values, palette='Set2')
plt.title("Distribution of Content by Type on Netflix")
plt.ylabel("Number of Titles")
plt.xlabel("Type")
plt.show()

top_countries = df['country'].value_counts().head(10)

plt.figure(figsize=(10, 6))
sns.barplot(y=top_countries.index, x=top_countries.values, palette='coolwarm')
plt.title("Top 10 Countries with Most Netflix Content")
plt.xlabel("Number of Titles")
plt.ylabel("Country")
plt.show()

rating_counts = df['rating'].value_counts()

plt.figure(figsize=(12, 5))
sns.barplot(x=rating_counts.index, y=rating_counts.values, palette='magma')
plt.title("Distribution of Netflix Content by Rating")
plt.xticks(rotation=45)
plt.xlabel("Rating")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 5))
sns.countplot(data=df, x='year_added', order=sorted(df['year_added'].dropna().unique()), palette='viridis')
plt.title("Number of Titles Added Each Year")
plt.xticks(rotation=45)
plt.xlabel("Year Added")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(12, 4))
sns.countplot(data=df, x='month_added', palette='cubehelix')
plt.title("Monthly Distribution of Netflix Releases")
plt.xlabel("Month")
plt.ylabel("Count")
plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.show()

from collections import Counter

# Split and flatten genre lists
all_genres = df['listed_in'].str.split(',').sum()
genre_counts = Counter([genre.strip() for genre in all_genres])
top_genres = pd.Series(genre_counts).sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_genres.values, y=top_genres.index, palette='Set3')
plt.title("Top 10 Genres on Netflix")
plt.xlabel("Count")
plt.ylabel("Genre")
plt.show()

top_directors = df['director'].value_counts().drop('Unknown').head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_directors.values, y=top_directors.index, palette='Blues_d')
plt.title("Top 10 Directors on Netflix")
plt.xlabel("Number of Titles")
plt.ylabel("Director")
plt.show()

from wordcloud import WordCloud

titles = ' '.join(df['title'].dropna())
wordcloud = WordCloud(width=800, height=400, background_color='black').generate(titles)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Netflix Titles", fontsize=18)
plt.show()

# Load the cleaned data
df = pd.read_csv("netflix_cleaned.csv")

# Fill any missing data in title or listed_in (shouldn't happen if cleaned properly)
df['title'] = df['title'].fillna("")
df['listed_in'] = df['listed_in'].fillna("")

# Create a combined feature for TF-IDF
df['content'] = df['title'] + " " + df['listed_in']
# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['content'])

def recommend_titles(title, df, tfidf_matrix, top_n=5):
    # Convert title to lowercase for better match
    title = title.lower()

    # Find the index of the movie that matches the title
    idx = df[df['title'].str.lower() == title].index

    if len(idx) == 0:
        return f"No match found for '{title.title()}' 😕"

    idx = idx[0]

    # Compute cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    # Get indices of top similar titles
    similar_indices = cosine_sim.argsort()[-top_n-1:-1][::-1]

    # Fetch recommended titles
    recommended = df.iloc[similar_indices][['title', 'type', 'listed_in', 'country']]

    return recommended

# Example usage
user_input = "The Crown"  # change this title
recommendations = recommend_titles(user_input, df, tfidf_matrix)

print(f"\nTop 5 recommendations for '{user_input}':\n")
print(recommendations)
