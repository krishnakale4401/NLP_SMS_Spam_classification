from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_vectorize(texts):
    """
    Convert text into Vectors using TF-IDF
    """

    ## Initialize the TFIDF
    vectorizer = TfidfVectorizer(max_features=5000) ## Limit to top 5000 features

    X = vectorizer.fit_transform(texts)

    return X, vectorizer