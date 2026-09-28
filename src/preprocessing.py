import re   ## Regular expressions for text preprocessing
import nltk
from nltk.corpus import stopwords    ## Load Stopwords for text preprocessing and filter less informative words
from nltk.stem import WordNetLemmatizer ## Lemmatization for reducing words to their base form

## Download NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english")) ## Set of stopwords for filtering

lemmatizer = WordNetLemmatizer() ## Initialize the WordNet Lemmatizer

def clean_text(text):
    """
    Clean and preprocess the input text converting it to:
    1. Lower case
    2. Remove URL's
    3. Remove special characters
    4. Remove stopwords and lemmatize
    """
    ## Convert to lowercase
    text = text.lower()
    ## Remove URLs
    text = re.sub(r"http\S+", "", text)  ## https://abc.com -> removed
    ## Remove special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  ## Keep letters and numbers
    
    ## Split into words
    words = text.split()

    ## Remove stopwords and lemmatize
    words = [lemmatizer.lemmatize(i) for i in words if i not in stop_words]
    
    ## Join back into sentence
    return " ".join(words)