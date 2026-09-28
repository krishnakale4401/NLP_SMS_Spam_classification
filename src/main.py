import pandas as pd

from src.preprocessing import clean_text
from src.vectorization import tfidf_vectorize
from src.analysis import get_top_spam_words



def run_pipeline():
    df = pd.read_csv("./data/spam.csv", encoding="latin-1")

    df = df[["v1", "v2"]]
    df.columns = ["label", "text"]

    ## Encode the labels
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

    # Preprocess the text to get the cleaned version with no URLs, special characters, or stopwords
    print("Cleaning text...")
    df["clean_text"] = df["text"].apply(clean_text)

    ## Show the cleaned text
    print("Cleaned text:")
    print(df["clean_text"].head())

    print("TF-IDF Vectorization:..")
    X, vectorizer = tfidf_vectorize(df["clean_text"])

    print("TF-IDF Vectors:")
    print(X.toarray())

    print("\n Top Spam words:...")
    spam_words = get_top_spam_words(vectorizer, X, df['label_num'], 1)

    for word, score in spam_words:
        print(f"Word: {word}, Score: {score}")

    print("\n Top Ham words:...")
    ham_words = get_top_spam_words(vectorizer, X, df['label_num'], 0)

    for word, score in ham_words:
        print(f"Word: {word}, Score: {score}")

    print("\n Pipeline complete")

if __name__ == "__main__":
    run_pipeline()