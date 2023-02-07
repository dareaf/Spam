import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import requests
import pickle
import joblib
from stop_words import get_stop_words

stop_words = get_stop_words('english')
stopwords = set(stop_words)

def isalpha(string):
    string = string.replace('.', '')
    return string.isalpha()

def clean_sms(string):
    string = string.lower()
    return (' '.join(filter(lambda x: isalpha(x) and x not in stopwords, string.split()))).replace('.', '').split()

gmm = joblib.load('./finalized_model.sav')
cv = pickle.load(open('./vectorizer.pkl', 'rb'))

def find_digit_percent(string):
    count = 0
    for i in string:
        if i.isdigit():
            count += 1
    return count / (len(string) + 1)

# function to find percentage of question marks in a string
def find_question_percent(string):
    count = 0
    for i in string:
        if i == '?':
            count += 1
    return count / (len(string) + 1)

# function to find percentage of exclamation marks in a string
def find_exclamation_percent(string):
    count = 0
    for i in string:
        if i == '!':
            count += 1
    return count / (len(string) + 1)

# function to find percentage of capital letters in a string
def find_capital_percent(string):
    count = 0
    for i in string:
        if i.isupper():
            count += 1
    return count / (len(string) + 1)

# function to find percentage of special characters in a string
def find_special_percent(string):
    count = 0
    for i in string:
        if not i.isalnum():
            count += 1
    return count / (len(string) + 1)

# function to if a string contains a emoji
def find_emoji(string):
    return int(':)' in string or ':(' in string or ':-)' in string or ':=D' in string or ':D' in string or ':P' in string)


def test_message(message):
    message = pd.Series(message)
    # convert the series to a dataframe
    message = pd.DataFrame(message, columns=['v2'])
    message['digit_percent'] = message['v2'].apply(find_digit_percent)
    message['question_percent'] = message['v2'].apply(find_question_percent)
    message['exclamation_percent'] = message['v2'].apply(find_exclamation_percent)
    message['capital_percent'] = message['v2'].apply(find_capital_percent)
    message['special_percent'] = message['v2'].apply(find_special_percent)
    message['emoji'] = message['v2'].apply(find_emoji)
    message = pd.concat([message, pd.DataFrame(cv.transform(message['v2']).toarray(), columns=cv.get_feature_names())], axis=1)
    
    X_test = message.drop(['v2'], axis=1)

    #st.dataframe(X_test)

    return gmm.predict(X_test), gmm.predict_proba(X_test)

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def main():

    st.set_page_config(page_title="Spam Detection", page_icon="mailbox_with_mail", layout="wide", initial_sidebar_state="expanded")

    option = option_menu(
        menu_title = None,
        options = ["Home", "Detector"],
        icons = ["house", "gear"],
        menu_icon = 'cast',
        default_index = 0,
        orientation = "horizontal"
    )

    if option == "Home":
        st.markdown("<h1 style='text-align: center; color: #08e08a;'>Spam Detection</h1>", unsafe_allow_html=True)
        lottie_hello = load_lottieurl("https://assets6.lottiefiles.com/packages/lf20_tuzu65Bu6N.json")

        st_lottie(
            lottie_hello,
            speed=1,
            reverse=False,
            loop=True,
            quality="High",
            #renderer="svg",
            height=400,
            width=-900,
            key=None,
        )

        st.markdown("<h4 style='text-align: center; color: #dae5e0;'>Enter your text and let the model decide wheater it's spam or ham</h4>", unsafe_allow_html=True)

    elif option == "Detector":
        st.markdown("<h1 style='text-align: center; color: #08e08a;'>Spam Detector</h1>", unsafe_allow_html=True)
        text = st.text_area("Enter your text here")

        if st.button("Detect"):
            result = test_message(text)
            
            # write probability

            if result[0][0] == 0:
                st.success("Yahoo! it's a HAM")
                st.balloons()
            elif result[0][0] == 1:
                st.error("Booo! it's a SPAM")


if __name__ == "__main__":
    main()