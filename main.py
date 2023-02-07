# importing libraries
import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import requests
import pickle
import joblib
from stop_words import get_stop_words

stop_words = get_stop_words('english') # list of english stopwords
stopwords = set(stop_words) # set of english stopwords

def isalpha(string): # function to check if a string is alphabetic
    string = string.replace('.', '') # remove full stops
    return string.isalpha() # return true if string is alphabetic

def clean_sms(string): # function to clean a string
    string = string.lower() # convert string to lowercase
    return (' '.join(filter(lambda x: isalpha(x) and x not in stopwords, string.split()))).replace('.', '').split() # return a list of words in the string after removing stopwords

gmm = joblib.load('./finalized_model.sav') # load the model
cv = pickle.load(open('./vectorizer.pkl', 'rb')) # load the vectorizer

def find_digit_percent(string): # function to find percentage of digits in a string
    count = 0 # initialize count to 0 
    for i in string: # iterate through each character in the string
        if i.isdigit(): # if the character is a digit
            count += 1 # increment count by 1 
    return count / (len(string) + 1)    # return the percentage of digits in the string

# function to find percentage of question marks in a string
def find_question_percent(string): # function to find percentage of question marks in a string
    count = 0 # initialize count to 0
    for i in string: # iterate through each character in the string
        if i == '?': # if the character is a question mark
            count += 1 # increment count by 1
    return count / (len(string) + 1) # return the percentage of question marks in the string

# function to find percentage of exclamation marks in a string
def find_exclamation_percent(string): # function to find percentage of exclamation marks in a string
    count = 0 # initialize count to 0
    for i in string: # iterate through each character in the string
        if i == '!': # if the character is an exclamation mark
            count += 1 # increment count by 1
    return count / (len(string) + 1) # return the percentage of exclamation marks in the string

# function to find percentage of capital letters in a string
def find_capital_percent(string): # function to find percentage of capital letters in a string
    count = 0 # initialize count to 0
    for i in string: # iterate through each character in the string
        if i.isupper():     # if the character is a capital letter
            count += 1 # increment count by 1
    return count / (len(string) + 1) # return the percentage of capital letters in the string

# function to find percentage of special characters in a string
def find_special_percent(string): # function to find percentage of special characters in a string
    count = 0 # initialize count to 0
    for i in string: # iterate through each character in the string
        if not i.isalnum(): # if the character is not alphanumeric
            count += 1 # increment count by 1
    return count / (len(string) + 1) # return the percentage of special characters in the string

# function to if a string contains a emoji
def find_emoji(string): # function to if a string contains a emoji
    return int(':)' in string or ':(' in string or ':-)' in string or ':=D' in string or ':D' in string or ':P' in string) # return 1 if the string contains a emoji else 0


def test_message(message): # function to test a message 
    message = pd.Series(message) # convert the message to a series 
    # convert the series to a dataframe 
    message = pd.DataFrame(message, columns=['v2']) # convert the series to a dataframe
    message['digit_percent'] = message['v2'].apply(find_digit_percent) # find the percentage of digits in the message
    message['question_percent'] = message['v2'].apply(find_question_percent) # find the percentage of question marks in the message
    message['exclamation_percent'] = message['v2'].apply(find_exclamation_percent) # find the percentage of exclamation marks in the message
    message['capital_percent'] = message['v2'].apply(find_capital_percent) # find the percentage of capital letters in the message
    message['special_percent'] = message['v2'].apply(find_special_percent) # find the percentage of special characters in the message
    message['emoji'] = message['v2'].apply(find_emoji) # find if the message contains a emoji
    message = pd.concat([message, pd.DataFrame(cv.transform(message['v2']).toarray(), columns=cv.get_feature_names())], axis=1) # convert the message to a vector and add it to the dataframe 
    
    X_test = message.drop(['v2'], axis=1) # drop the message column from the dataframe

    #st.dataframe(X_test)

    return gmm.predict(X_test), gmm.predict_proba(X_test)   # return the prediction and the probability of the prediction

def load_lottieurl(url: str): # function to load a lottie animation from a url
    r = requests.get(url) # get the url
    if r.status_code != 200: # if the url is not valid
        return None # return None
    return r.json() # return the json file

def main(): # main function

    st.set_page_config(page_title="Spam Detection", page_icon="mailbox_with_mail", layout="wide", initial_sidebar_state="expanded")  # set the page config

    option = option_menu( # create a option menu
        menu_title = None, # set the menu title to None
        options = ["Home", "Detector"], # set the options to Home and Detector
        icons = ["house", "gear"], # set the icons for the options
        menu_icon = 'cast', # set the menu icon
        default_index = 0, # set the default index to 0
        orientation = "horizontal" # set the orientation to horizontal
    )

    if option == "Home": # if the option is Home
        st.markdown("<h1 style='text-align: center; color: #08e08a;'>Spam Detection</h1>", unsafe_allow_html=True) # set the title to Spam Detection
        lottie_hello = load_lottieurl("https://assets6.lottiefiles.com/packages/lf20_tuzu65Bu6N.json") # load the lottie animation from the url

        st_lottie( # display the lottie animation
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

        st.markdown("<h4 style='text-align: center; color: #dae5e0;'>Enter your text and let the model decide wheater it's spam or ham</h4>", unsafe_allow_html=True) # set the text to Enter your text and let the model decide wheater it's spam or ham

    elif option == "Detector": # if the option is Detector
        st.markdown("<h1 style='text-align: center; color: #08e08a;'>Spam Detector</h1>", unsafe_allow_html=True) # set the title to Spam Detector
        text = st.text_area("Enter your text here") # create a text area to enter the text

        if st.button("Detect"): # if the detect button is clicked
            result = test_message(text) # test the message
            
            # write probability

            if result[0][0] == 0: # if the result is 0
                st.success("Yahoo! it's a HAM") # display a success message
                st.balloons() # display balloons
            elif result[0][0] == 1: # if the result is 1
                st.error("Booo! it's a SPAM") # display an error message


if __name__ == "__main__": # if the file is run directly
    main() # run the main function