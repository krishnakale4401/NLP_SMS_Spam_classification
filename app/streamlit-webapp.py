import sys  ## Controls all the python runtime (e.g. imports)
import os   ## Provides a way of working with files and directories

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  ## Adds the parent directory to the system path

## D:\GenAI\IntelliBI\GenAI\Class\Projects\NLP\Spam detecton using SMS\app\streamlit-webapp.py
## __file__ : Give you current file location
## os.path.dirname(__file__) : Gives you the directory name of the current file
## .. : Go one folder above
## os.path.join() : Joins one or more path components
## os.path.abspath() : Returns the absolute path of a file

import streamlit as st
from src.preprocessing import clean_text

SPAM_KEYWORDS = ["free", "win", "winner", "cash", "prize", "money", "offer", "click", "buy", "now", "limited time"]

def rule_based_prediction(text):
    text = clean_text(text)

    score = 0
    matched = []

    for word in SPAM_KEYWORDS:
        if word in text:
            score += 1
            matched.append(word)

    if score > 1:
        return "Spam", score, matched
    else:
        return "Ham", score, matched

st.title("NLP Spam Detection")
st.write("Enter a message to analyze if it's spam or ham (not spam).")

user_input = st.text_area("Enter your message:")   

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter a message to analyze.")
    else:
        label, score, matched = rule_based_prediction(user_input)

        if label == "Spam":
            st.error("Spam message detected!")
        else:
            st.success("Ham message detected.")
        
        st.write(f"Spam Score: {score}")
        st.write(f"Matched Keywords: {', '.join(matched)}")