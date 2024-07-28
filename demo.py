import streamlit as st
import nltk
import os
import re
import pandas as pd
#from langchain.llms import GooglePalm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from dotenv import load_dotenv
import google.generativeai as gen_ai
from nltk.stem import PorterStemmer
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
stop_words=set(stopwords.words('english'))
stemmer = PorterStemmer()
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# Set up Google Gemini-Pro AI model
gen_ai.configure(api_key=GOOGLE_API_KEY)
model = gen_ai.GenerativeModel('gemini-pro')
disease=[]
remedies=[]
symptoms=[]
how=[]
#collecting data from csv
#def load_data(file_path):
file_path = r'C:\Users\ANWESHA\OneDrive\Documents\python\Python Project\ayurvedic_QnA.csv'
data = pd.read_csv(file_path)
    

for index, row in data.iterrows():
    how.append(row['How to Apply Remedies'])
    disease.append(row['Disease'])
    remedies.append(row['Remedies'])
    symptoms.append(row['Symptoms'].split(','))
    
    # for symptom in symptoms:
    #         symptom = symptom.strip()
    #         question = symptom
    #         answer = f"Based on the problem you are facing, you have {disease}.\n The Remedies to cure the disease are as follows: {remedies}.\n   {row['How to Apply Remedies']}"
    #         questions.append(question)
    #         answers.append(answer)

    


#load_data(file_path)
print(disease)

# Function to translate roles between Gemini-Pro and Streamlit terminology
def translate_role_for_streamlit(user_role):
    if user_role == "model":
        return "assistant"
    else:
        return "HI anwesha"





def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

problem = [
    "Cough", 
    "Cold", 
    "Head ache", 
    "Indigestion", 
    "Acidity", 
    "Constipation", 
    "Joint Pain", 
    "Insomnia", 
    "Anxiety", 
    "Skin Rash", 
    "Hair Loss", 
    "Sore Throat", 
    "Fatigue", 
    "Back Pain", 
    "High Blood Pressure", 
    "Diabetes", 
    "Allergies", 
    "Menstrual Cramps", 
    "Weight Loss", 
    "Depression","Hi","hello","What are you doing?","thank you"
]

solution = [
    "Mix honey and ginger juice, take twice daily.",
    "Boil tulsi leaves and ginger in water, drink as tea.",
    "Apply a paste of ginger powder and water on the forehead.",
    "Drink warm water with a teaspoon of cumin seeds.",
    "Consume a mixture of aloe vera juice and honey.",
    "Drink warm water with lemon juice and honey in the morning.",
    "Apply a paste of turmeric and ginger on the affected area.",
    "Drink warm milk with a pinch of nutmeg powder before bed.",
    "Drink a tea made from ashwagandha and brahmi leaves.",
    "Apply a paste of neem and turmeric on the rash.",
    "Massage the scalp with warm coconut oil mixed with hibiscus powder.",
    "Gargle with warm salt water mixed with turmeric.",
    "Drink ashwagandha tea to boost energy levels.",
    "Apply a paste of ginger powder and eucalyptus oil on the back.",
    "Drink a tea made from holy basil and lemon balm.",
    "Consume a mixture of bitter gourd juice and turmeric.",
    "Drink a tea made from licorice root and turmeric.",
    "Drink a tea made from fennel seeds and ginger.",
    "Drink a mixture of honey and lemon juice in warm water.",
    "Consume a tea made from saffron and turmeric.","hi i am 'AyurMind Bot' how can i help you","i am 'AyurMind  Bot' and can assist you in ayurvedic remedies","i am ai chat bot doing fine ",
    "You are most welcomed from me 'AyurMind Bot'"
]

st.set_page_config(
    page_title="Chat with AyurMind Bot!",
    page_icon=":brain:",  # Favicon emoji
    layout="centered",  # Page layout option
)
# Define your custom responses and corresponding Markdown text
# responses = {
#     "hi": "Hello! How can I assist you today?",
#     "bye": "Goodbye! Have a great day!",
#     "info": "This is a Streamlit chatbot example. You can ask questions and get responses.",
# }


# Streamlit app
#Title and desc
st.title("AyurMind Bot")
st.write("Welcome to AyurMind Bot!")


#User input
user_input = st.text_input("Enter your query here:", "")
max2=0
c=0
index=0
#Function to respond to user input
def bot_response(user_input):
    for i in disease:
        print(i)
        global c
        text1=preprocess_text(i)
        text2=preprocess_text(user_input)
        text3=preprocess_text(symptoms[c])
        vectorizer=TfidfVectorizer()
        X=vectorizer.fit_transform([text1,text2]) # Vectorizing
        X2=vectorizer.fit_transform([text3,text2]) # Vectorizing
        print(X)
        cosine_sim=cosine_similarity(X[0],X[1])[0][0]
        cosine_sim2=cosine_similarity(X[0],X[1])[0][0]
        print(cosine_sim)
        
        match_prob=max(cosine_sim*100,cosine_sim2*100)
        global max2
        if match_prob>max2:
            
            max2=match_prob
            global index
            index=c 
            #print(index)
        c+=1
    #print(match_prob)
    if max2<20:
        # my_prompt=f"""provide analysis of "{i}"
        # """
        # output=llm2(my_prompt)
        return("Sorry! Can't provide a solution")
    else:
        return(f"elaborate the solution '{remedies[index]}' in 2 sentences and also add '{how[index]}' after")
    

# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = model.start_chat(history=[])

if "history" not in st.session_state:
    st.session_state.history = []




def add_to_chat_history(user_msg, bot_msg):
    st.session_state.history.append({"user": user_msg, "bot": bot_msg})
#Button for submitting response
#if st.button("Ask"):
if user_input:    
    #print(st.session_state.chat_history)
    bot_reply=bot_response(user_input)
    #user_pro="'"+bot_reply+"'"+" elaborate it in 4 sentences"
    gemini_response = model.generate_content(bot_reply).text
    st.text_area("Bot's response:",value=gemini_response,height=100)
    #st.chat_message("user").markdown(user_input)
    for chat in st.session_state.history:
        st.chat_message("user").markdown(chat['user'])
        st.chat_message("bot").markdown(chat['bot'])
    #     st.markdown(f"**User:** {chat['user']}")
    #     st.markdown(f"**Bot:** {chat['bot']}")
    add_to_chat_history(user_input,gemini_response)


# for message in st.session_state.chat_history.history:
#     with st.chat_message(translate_role_for_streamlit(message.role)):
#         st.markdown(user_input)
#         st.markdown(gemini_response)



