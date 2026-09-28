import numpy as np

def get_top_spam_words(vectorizer, X, label_series, label_value, top_n = 20):
    """
    Get the top n spam words for spam or ham
    vectorizer = words
    X = TF-IDF matrix
    label_series = labels of spam/ham
    label_value = 1 for spam, 0 for ham
    """
    ## create a boolean mask for the desired label
    mask = (label_series == label_value).values
    print(f"Number of samples for label {label_value}: {mask.sum()}")

    ## Filter rows
    X_filtered = X[mask]

    ## Calculate the importance of each word
    words = vectorizer.get_feature_names_out()

    ## Calculate the average TF-IDF score for each word
    avg_tfidf = X_filtered.mean(axis=0).A1

    ## AI is a shortcut in Numpy for converting output to 1D array

    ## Pair all the words with their average TF-IDF score
    word_scores = list(zip(words, avg_tfidf))

    ## Sort by score based on importance
    sorted_words = sorted(word_scores, key=lambda x: x[1], reverse=True)

    return sorted_words[:top_n]