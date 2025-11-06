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
# This is now MUCH smaller because min_df, max_df, 
# and token_pattern do most of the work.
# =================================================================
stop_words_list = list(nltk.corpus.stopwords.words('english'))
custom_junk = [
    'http', 'https', 'com', 'www', 'youtu', 'be', 'ly', 'goo', 'gl', # URLs
]
stop_words_list.extend(custom_junk)
CUSTOM_STOP_WORDS = set(stop_words_list)
# =================================================================


def preprocess_text(text: str) -> str:
    """Cleans text for TF-IDF."""
    if not isinstance(text, str):
        return ""
    text = text.lower()  # Lowercase
    text = re.sub(r'\S+@\S+', ' ', text)  # Remove emails
    text = re.sub(r'http\S+', ' ', text)  # Remove URLs
    
    # --- ADDED FROM FRIEND'S SUGGESTION 2 ---
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)   # Split camelCase (e.g., VisionPro)
    text = re.sub(r'([a-z])(\d)', r'\1 \2', text)      # Split words with numbers (e.g., M5)
    
    text = re.sub(r'[^a-z\s]', ' ', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace

    return text.strip()

def create_fingerprint(video_df: pd.DataFrame, top_n: int = 15) -> list[str]:
    """
    Generates a "fingerprint" (top TF-IDF keywords) for a channel.
    This version is fully robust with min/max df and token patterns.
    """
    if video_df.empty:
        return []

    #With titles and descriptions 
    corpus = []
    for _, row in video_df.iterrows():
        
        title = str(row.get('title', '')) * 3 # Triple title weight
        description = str(row.get('description', ''))
        
        doc_text = title + ' ' + description
        clean_doc = preprocess_text(doc_text)
        
        if clean_doc:
            corpus.append(clean_doc)


    if len(corpus) < 2:
        print(f"Warning: Only found {len(corpus)} videos. min_df=2 will filter everything.")
        # We can either return [] or rerun with min_df=1
        # For now, let's just return empty.
        return []

    vectorizer = TfidfVectorizer(
            stop_words=list(CUSTOM_STOP_WORDS),
            max_features=1000,
            
            min_df=2,   # Ignore words that appear in < 2 videos (kills one-hit wonders)
            max_df=0.8, # Ignore words in > 80% of videos (kills boilerplate)
            
            token_pattern=r'\b[a-z]{3,}\b' # Only accept words 3+ letters long
        )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        # because min_df=2 will filter almost everything. This is OK.
        print("No features found. This is common with min_df=2 on small samples.")
        return []

    average_scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
    feature_names = vectorizer.get_feature_names_out()
    
    df_scores = pd.DataFrame({'keyword': feature_names, 'score': average_scores})
    
    top_keywords = df_scores.sort_values(by='score', ascending=False).head(top_n)
    
    print("\n--- Top keywords and scores (robust) ---")
    print(top_keywords.to_string())
    
    return list(top_keywords['keyword'])