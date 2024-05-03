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
