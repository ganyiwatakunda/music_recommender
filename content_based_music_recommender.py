# -*- coding: utf-8 -*-
"""content_based_music_recommender.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1HTuoKYSGJRkbE3H9a_NOxAe1-1Yz2s4S

# Music recommender system

One of the most used machine learning algorithms is recommendation systems. A **recommender** (or recommendation) **system** (or engine) is a filtering system which aim is to predict a rating or preference a user would give to an item, eg. a film, a product, a song, etc.

Which type of recommender can we have?   

There are two main types of recommender systems:
- Content-based filters
- Collaborative filters
  
> Content-based filters predicts what a user likes based on what that particular user has liked in the past. On the other hand, collaborative-based filters predict what a user like based on what other users, that are similar to that particular user, have liked.

### 1) Content-based filters

Recommendations done using content-based recommenders can be seen as a user-specific classification problem. This classifier learns the user's likes and dislikes from the features of the song.

The most straightforward approach is **keyword matching**.

In a few words, the idea behind is to extract meaningful keywords present in a song description a user likes, search for the keywords in other song descriptions to estimate similarities among them, and based on that, recommend those songs to the user.

*How is this performed?*

In our case, because we are working with text and words, **Term Frequency-Inverse Document Frequency (TF-IDF)** can be used for this matching process.
  
We'll go through the steps for generating a **content-based** music recommender system.

### Importing required libraries

First, we'll import all the required libraries.
"""

import numpy as np
import pandas as pd

from typing import List, Dict

"""We have already used the **TF-IDF score before** when performing Twitter sentiment analysis.

Likewise, we are going to use TfidfVectorizer from the Scikit-learn package again.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

"""### Dataset

So imagine that we have the [following dataset](https://www.kaggle.com/mousehead/songlyrics/data#).

This dataset contains name, artist, and lyrics for *57650 songs in English*. The data has been acquired from LyricsFreak through scraping.
"""

# Commented out IPython magic to ensure Python compatibility.
from google.colab import drive
drive.mount('/gdrive')
# %cd /gdrive

songs = pd.read_csv('/gdrive/MyDrive/songdata.csv')

songs = pd.read_csv('songdata.csv')

songs.head()

"""Because of the dataset being so big, we are going to resample only 5000 random songs."""

songs = songs.sample(n=5000).drop('link', axis=1).reset_index(drop=True)

"""We can notice also the presence of `\n` in the text, so we are going to remove it."""

songs['text'] = songs['text'].str.replace(r'\n', '')

"""After that, we use TF-IDF vectorizerthat calculates the TF-IDF score for each song lyric, word-by-word.

Here, we pay particular attention to the arguments we can specify.
"""

tfidf = TfidfVectorizer(analyzer='word', stop_words='english')

lyrics_matrix = tfidf.fit_transform(songs['text'])

"""*How do we use this matrix for a recommendation?*

We now need to calculate the similarity of one lyric to another. We are going to use **cosine similarity**.

We want to calculate the cosine similarity of each item with every other item in the dataset. So we just pass the lyrics_matrix as argument.
"""

cosine_similarities = cosine_similarity(lyrics_matrix)

"""Once we get the similarities, we'll store in a dictionary the names of the 50  most similar songs for each song in our dataset."""

similarities = {}

for i in range(len(cosine_similarities)):
    # Now we'll sort each element in cosine_similarities and get the indexes of the songs.
    similar_indices = cosine_similarities[i].argsort()[:-50:-1]
    # After that, we'll store in similarities each name of the 50 most similar songs.
    # Except the first one that is the same song.
    similarities[songs['song'].iloc[i]] = [(cosine_similarities[i][x], songs['song'][x], songs['artist'][x]) for x in similar_indices][1:]

"""After that, all the magic happens. We can use that similarity scores to access the most similar items and give a recommendation.

For that, we'll define our Content based recommender class.
"""

class ContentBasedRecommender:
    def __init__(self, matrix):
        self.matrix_similar = matrix

    def _print_message(self, song, recom_song):
        rec_items = len(recom_song)

        print(f'The {rec_items} recommended songs for {song} are:')
        for i in range(rec_items):
            print(f"Number {i+1}:")
            print(f"{recom_song[i][1]} by {recom_song[i][2]} with {round(recom_song[i][0], 3)} similarity score")
            print("--------------------")

    def recommend(self, recommendation):
        # Get song to find recommendations for
        song = recommendation['song']
        # Get number of songs to recommend
        number_songs = recommendation['number_songs']
        # Get the number of songs most similars from matrix similarities
        recom_song = self.matrix_similar[song][:number_songs]
        # print each item
        self._print_message(song=song, recom_song=recom_song)

"""Now, instantiate class"""

recommedations = ContentBasedRecommender(similarities)

"""Then, we are ready to pick a song from the dataset and make a recommendation."""

recommendation = {
    "song": songs['song'].iloc[10],
    "number_songs": 4
}

recommedations.recommend(recommendation)

"""And we can pick another random song and recommend again:"""

recommendation2 = {
    "song": songs['song'].iloc[120],
    "number_songs": 4
}

recommedations.recommend(recommendation2)

recommendation3 = {
    "song": songs['song'].iloc[40],
    "number_songs": 4
}

recommedations.recommend(recommendation3)



"""# Streamlit App Testing

"""

!pip install streamlit

import streamlit as st

def main():
    st.title("Music Recommendation System")

    # Sidebar input
    song_title = st.sidebar.text_input("Enter a song title")
    number_songs = st.sidebar.slider("Number of songs to recommend", min_value=1, max_value=10, value=5)

    if st.sidebar.button("Recommend"):
        # Create an instance of the ContentBasedRecommender class
        recommender = ContentBasedRecommender(cosine_similarities)
        # Define the recommendation input
        recommendation = {'song': song_title, 'number_songs': number_songs}
        # Get recommendations
        recommendations = recommender.recommend(recommendation)

if __name__ == '__main__':
    main()

