import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import re
import numpy as np 

# Download the NLTK stopwords list (only needs to run once)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')

# Create a custom stop words list
# to remove junk
# =================================================================
stop_words_list = list(nltk.corpus.stopwords.words('english'))
custom_junk = [
    'http', 'https', 'com', 'www', 'youtu', 'be', 'ly', 'goo', 'gl', # URLs
    'subscribe', 'channel', 'video', 'videos', 'watch', 'check',     # YouTube actions
    'rmb', 'usd', 'price',                                           # Common video junk
    'one', 'also', 'first', 'make', 'take', 'back', 'get', 'try',    # Common english junk
    'go', 'come', 'day', 'night', 'street', 'tour'                   # Too generic
]
stop_words_list.extend(custom_junk)
# Convert to a set for faster lookups
CUSTOM_STOP_WORDS = set(stop_words_list)
# =================================================================


def preprocess_text(text: str) -> str:
    """Cleans text for TF-IDF."""
    if not isinstance(text, str):
        return ""
    text = text.lower()  # Lowercase
    text = re.sub(r'\S+@\S+', ' ', text)  # Remove emails
    text = re.sub(r'http\S+', ' ', text)  # Remove URLs
    text = re.sub(r'[^a-z\s]', ' ', text)  # Remove punctuation/numbers
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    return text.strip()

def create_fingerprint(video_df: pd.DataFrame, top_n: int = 15) -> list[str]:
    """
    Generates a "fingerprint" (top TF-IDF keywords) for a channel.
    
    This NEW version treats EACH video as a separate document,
    then averages the TF-IDF scores to find the true channel fingerprint.
    """
    if video_df.empty:
        return []

    # Create a "corpus" (list of documents), one for each video.
    # =================================================================
    corpus = []
    for _, row in video_df.iterrows():
        # Give title more weight
        title = str(row.get('title', '')) * 2
        description = str(row.get('description', ''))
        
        doc_text = title + ' ' + description
        clean_doc = preprocess_text(doc_text)
        
        if clean_doc:
            corpus.append(clean_doc)

    if not corpus:
        print("No text left after preprocessing. Skipping fingerprint.")
        return []
    # =================================================================

    # 3. Use TF-IDF to find top keywords
    # pass our new custom stop words list
    vectorizer = TfidfVectorizer(stop_words=list(CUSTOM_STOP_WORDS), max_features=1000)
    
    try:
        tfidf_matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        return []

    # Average the scores across all documents (videos)
    # =================================================================
    # tfidf_matrix is (num_videos, num_keywords)
    # want the average score for each keyword
    average_scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
    # =================================================================

    # Get feature names (the keywords)
    feature_names = vectorizer.get_feature_names_out()
    
    # Create a DataFrame of keywords and their *average* scores
    df_scores = pd.DataFrame({'keyword': feature_names, 'score': average_scores})
    
    # 4. Get the top N keywords
    top_keywords = df_scores.sort_values(by='score', ascending=False).head(top_n)
    
    print("\n--- Top keywords and scores (post-averaging) ---")
    print(top_keywords.to_string()) # Print the top keywords and their scores
    
    return list(top_keywords['keyword'])