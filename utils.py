import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

def load_data():
    df = pd.read_csv("research_papers.csv")
    df['cleaned'] = df['abstract'].apply(preprocess)
    return df

def vectorize(df):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['cleaned'])
    return vectorizer, tfidf_matrix

def search_papers(query, df, vectorizer, tfidf_matrix):
    query = preprocess(query)
    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, tfidf_matrix)
    scores = list(enumerate(similarity[0]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return scores

def get_answer(query, df, vectorizer, tfidf_matrix):
    scores = search_papers(query, df, vectorizer, tfidf_matrix)
    best_index = scores[0][0]

    text = df.iloc[best_index]['abstract']

    # Split and remove empty sentences
    sentences = [s.strip() for s in text.split('.') if s.strip() != ""]

    definition = sentences[0] if len(sentences) > 0 else "No definition available"

    explanation = sentences[1] if len(sentences) > 1 else (
        "This concept focuses on enabling systems to learn from data, identify patterns, and improve performance automatically without explicit programming."
    )

    application = sentences[2] if len(sentences) > 2 else (
        "Widely used in healthcare, recommendation systems, automation, finance, and intelligent decision-making systems."
    )

    answer = f"""
📌 Definition:
{definition}.

📖 Explanation:
{explanation}.

🚀 Applications:
{application}.
"""

    return answer