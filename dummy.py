import random
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import faiss
import os
import pickle
from dotenv import load_dotenv
import google.generativeai as gen_ai

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize stemmer
stemmer = PorterStemmer()

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Set up Google Gemini-Pro AI model
gen_ai.configure(api_key=GOOGLE_API_KEY)
model = gen_ai.GenerativeModel('gemini-pro')

# Preprocess text: tokenization, stemming, and removing stopwords
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

def load_data(file_path):
    data = pd.read_csv(file_path)
    questions = []
    answers = []

    for index, row in data.iterrows():
        disease = row['Disease']
        remedies = row['Remedies']
        symptoms = row['Symptoms'].split(',')

        for symptom in symptoms:
            symptom = symptom.strip()
            question = symptom
            answer = f"Based on the problem you are facing, you have {disease}.\n The Remedies to cure the disease are as follows: {remedies}.\n   {row['How to Apply Remedies']}"
            questions.append(question)
            answers.append(answer)

    return questions, answers

# Example usage
file_path = r'C:\Users\ANWESHA\OneDrive\Documents\python\Python Project\ayurvedic_QnA.csv'
questions, answers = load_data(file_path)

# Function to save chat history to file
# def save_chat_history(chat_history, file_name="chat_history.pkl"):
#     with open(file_name, "wb") as file:
#         pickle.dump(chat_history, file)

# # Function to load chat history from file
# def load_chat_history(file_name="chat_history.pkl"):
#     if os.path.exists(file_name):
#         with open(file_name, "rb") as file:
#             return pickle.load(file)
#     return []

# Streamlit App
def main():
    # Streamlit UI
    st.title(" 🤖 Welcome to Ayurmind Bot 🤖")

    # Initialize session state if it does not exist
    if 'chat_history' not in st.session_state:
        # st.session_state.chat_history = load_chat_history()
        st.session_state.chat_history = []

    # Initialize symptoms and solutions
    symptoms = questions
    solutions = answers

    # Preprocess the symptoms
    preprocessed_symptoms = [preprocess_text(symptom) for symptom in questions]

    # Create TF-IDF Vectorizer and fit on symptoms
    vectorizer = TfidfVectorizer()
    vectorizer.fit(preprocessed_symptoms)
    tfidf_matrix_symptoms = vectorizer.transform(preprocessed_symptoms)

    # Determine the dimension of TF-IDF vectors
    # d = len(vectorizer.get_feature_names_out())

    # Initialize FAISS index with the correct dimension
    # index = faiss.IndexFlatL2(d)

    user_input = st.text_input("What can Ayurveda help you with? Share your symptoms:")

    if user_input:
        # Preprocess user input
        preprocessed_input = preprocess_text(user_input)
        input_vector = vectorizer.transform([preprocessed_input])

        # Calculate cosine similarity for symptoms
        similarities_symptoms = cosine_similarity(input_vector, tfidf_matrix_symptoms)
        max_similarity_symptoms = np.max(similarities_symptoms)
        index_of_max_symptoms = np.argmax(similarities_symptoms)

        # Check if similarity is greater than 40%
        if max_similarity_symptoms > 0.4:
            # response = solutions[index_of_max_symptoms]
            response = model.generate_content(f"If you find that the prompt is a disease, keep it within 4 sentences, Otherwise if it is greeting or thanks then greet the user back, or write welcome as Ayurmind chatbot: {solutions[index_of_max_symptoms]}").text
        else:
        # elif max_similarity_symptoms < 0.2:
            response = model.generate_content(f"Sorry! Can't provide a solution").text
        # else:
        #     response = model.generate_content(f"Assume you are AyurMind chatbot Who provides cures based on ayurveda,now respond in that manner: {user_input}").text
        # Display responses and add to chat history
        st.session_state.chat_history.append({"user": user_input, "bot": response})
        # save_chat_history(st.session_state.chat_history)
        #st.markdown({"user": user_input})
        st.chat_message("AyurMind").markdown(response)
        # user_vector = input_vector.toarray().flatten().astype(np.float32)
        # response_vector = vectorizer.transform([preprocess_text(response)]).toarray().flatten().astype(np.float32)

        # index.add(np.array([user_vector, response_vector]))

    # Display chat history
    # for chat in st.session_state.chat_history:
    #     st.write(f"<div style='border: 1px solid #ddd; border-radius: 8px; padding: 10px; margin-bottom: 10px; background-color: white;'> <b> You:  </b> {chat['user']}</div>", unsafe_allow_html=True)
    #     st.write(f"<div style='border: 1px solid #ddd; border-radius: 8px; padding: 10px; margin-bottom: 10px; background-color: white;'><b> 🤖:  </b> {chat['bot']}</div>", unsafe_allow_html=True)

    # for chat in st.session_state.chat_history:
    #     st.markdown(f"<div style='border: 1px solid #ddd; border-radius: 8px; padding: 10px; margin-bottom: 10px; background-color: black;'> <b> You:  </b> {chat['user']}</div>", unsafe_allow_html=True)
    #     st.markdown(f"<div style='border: 1px solid #ddd; border-radius: 8px; padding: 10px; margin-bottom: 10px; background-color: black;'><b> 🤖:  </b> {chat['bot']}</div>", unsafe_allow_html=True)

    #Display chat history in the sidebar
    st.sidebar.title("Chat History")
    for chat in st.session_state.chat_history:
        st.sidebar.write(f"You: {chat['user']}")
        st.sidebar.write(f"🤖: {chat['bot']}")
        st.sidebar.write("---")

    # Add a bot icon to the text input
    st.markdown("""
        <style>
        * {
            background: black;
            color: white;
        }
        .stTextInput {
            padding-left: 40px;
               background: url('https://cdn.pixabay.com/photo/2023/03/05/21/11/ai-generated-7832244_640.jpg') no-repeat 10px center;
            background-size: 20px;
        }
        .stButton {
            text-align: center;
            width: auto;
            color: black;
            font-size: 16px;
            font-weight: bold;
            border-radius: 5px;
            padding: 10px;
        }
        .stButton:hover {
            font-size: 20px;
            font-weight: 650;
        }
        </style>
    """, unsafe_allow_html=True)


main()