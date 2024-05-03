import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
songs = pd.read_csv('content based recommedation system/songdata.csv')

# Create TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(songs['text'])

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Content-based recommender class
class ContentBasedRecommender:
    def __init__(self, matrix):
        self.matrix_similar = matrix

    def _print_message(self, song, recom_song):
        rec_items = len(recom_song)
        
        st.write(f'The {rec_items} recommended songs for {song} are:')
        for i in range(rec_items):
            st.write(f"Number {i+1}:")
            st.write(f"{recom_song[i][1]} by {recom_song[i][2]} with {round(recom_song[i][0], 3)} similarity score")
            st.write("--------------------")
        
    def recommend(self, recommendation):
        # Get song to find recommendations for
        song = recommendation['song']
        # Get number of songs to recommend
        number_songs = recommendation['number_songs']
        # Get the number of songs most similars from matrix similarities
        recom_song = self.matrix_similar[song][:number_songs]
        # Print each item
        self._print_message(song=song, recom_song=recom_song)

# Streamlit app
def main():
    st.title("Music Recommendation System")

    # Sidebar input
    song_title = st.sidebar.text_input("Enter a song title")
    number_songs = st.sidebar.slider("Number of songs to recommend", min_value=1, max_value=10, value=5)

    if st.sidebar.button("Recommend"):
        # Create an instance of the ContentBasedRecommender class
        recommender = ContentBasedRecommender(cosine_sim)
        # Define the recommendation input
        recommendation = {'song': song_title, 'number_songs': number_songs}
        # Get recommendations
        recommendations = recommender.recommend(recommendation)

if __name__ == '__main__':
    main()
