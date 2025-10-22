import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import re

# Download the NLTK stopwords list (only needs to run once)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')

# Load English stopwords
stop_words = list(nltk.corpus.stopwords.words('english'))

def preprocess_text(text: str) -> str:
    """Cleans text for TF-IDF."""
    text = text.lower()  # Lowercase
    text = re.sub(r'\S+@\S+', ' ', text)  # Remove emails
    text = re.sub(r'http\S+', ' ', text)  # Remove URLs
    text = re.sub(r'[^a-z\s]', ' ', text)  # Remove punctuation/numbers
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    return text.strip()

def create_fingerprint(video_df: pd.DataFrame, top_n: int = 15) -> list[str]:
    """
    Generates a "fingerprint" (top TF-IDF keywords) for a channel.
    
    Args:
        video_df: A DataFrame with 'title' and 'description' columns for *one* channel.
        top_n: The number of keywords to return.

    Returns:
        A list of the top N keywords.
    """
    if video_df.empty:
        return []

    # 1. Combine all titles and descriptions into one giant text block
    # We give titles more weight by repeating them (a simple trick)
    text_blob = ' '.join(video_df['title'] * 2) + ' ' + ' '.join(video_df['description'])
    
    # 2. Preprocess the text
    clean_text = preprocess_text(text_blob)
    
    if not clean_text:
        return []

    # 3. Use TF-IDF to find top keywords
    vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=1000)
    
    # We pass the text as a list (TF-IDF expects a collection of documents)
    try:
        tfidf_matrix = vectorizer.fit_transform([clean_text])
    except ValueError:
        # Happens if text is empty after preprocessing
        return []

    # Get feature names (the keywords)
    feature_names = vectorizer.get_feature_names_out()
    
    # Get the scores for our single document
    scores = tfidf_matrix.toarray()[0]
    
    # Create a DataFrame of keywords and their scores
    df_scores = pd.DataFrame({'keyword': feature_names, 'score': scores})
    
    print(df_scores.head(20))
    # 4. Get the top N keywords
    top_keywords = df_scores.sort_values(by='score', ascending=False).head(top_n)
    
    return list(top_keywords['keyword'])