import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv(
        r"C:\Users\shavani\Downloads\archive\zomato_pune_V002.csv").copy()  # Copy the DataFrame to avoid mutation warnings


data = load_data()

# Fill missing values and combine relevant information into a single feature
data.fillna("", inplace=True)
data['features'] = data[
    ['Restaurant_Name', 'Locality', 'Cuisines', 'Charges_for_two', 'Ratings_out_of_5', 'Number of votes']].apply(
    lambda x: ' '.join(x), axis=1)

# Initialize TfidfVectorizer to convert text data into numerical vectors
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the text data
tfidf_matrix = tfidf_vectorizer.fit_transform(data['features'])

# Compute similarity matrix
item_similarity = cosine_similarity(tfidf_matrix)


# Function to get recommendations for a restaurant
def get_item_recommendations(selected_option, selected_value, item_similarity, data=data, top_n=5):
    if selected_option == 'Restaurant Name':
        idx = data[data['Restaurant_Name'] == selected_value].index[0]
        accuracy = len(data[data['Restaurant_Name'] == selected_value]) / len(data)
    elif selected_option == 'Cuisine':
        idx = data[data['Cuisines'] == selected_value].index[0]
        accuracy = len(data[data['Cuisines'] == selected_value]) / len(data)
    elif selected_option == 'Location':
        idx = data[data['Locality'] == selected_value].index[0]
        accuracy = len(data[data['Locality'] == selected_value]) / len(data)

    sim_scores = list(enumerate(item_similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]
    restaurant_indices = [i[0] for i in sim_scores]
    return data.iloc[restaurant_indices], accuracy


# Streamlit app
st.title('Restaurant Recommendation System')

# User input in the main window
option = st.selectbox('Select an Option:', ['Restaurant Name', 'Cuisine', 'Location'])
if option == 'Restaurant Name':
    selected_value = st.selectbox('Select a Restaurant Name:', data['Restaurant_Name'].unique())
elif option == 'Cuisine':
    selected_value = st.selectbox('Select a Cuisine:', data['Cuisines'].unique())
elif option == 'Location':
    selected_value = st.selectbox('Select a Location:', data['Locality'].unique())

# Get recommendations and display them
if st.button('Get Recommendations'):
    recommendations, accuracy = get_item_recommendations(option, selected_value, item_similarity)
    st.subheader('Recommended Restaurants:')
    st.table(recommendations[
                 ['Restaurant_Name', 'Locality', 'Cuisines', 'Charges_for_two', 'Ratings_out_of_5', 'Number of votes']])
    st.write(f"Accuracy: {accuracy:.2f}")
