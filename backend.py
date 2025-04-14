import shelve
import re
from pathlib import Path
import os
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import math
from scipy.spatial.distance import cosine  # ✅ Use scipy cosine

class VectorSpaceQueryProcessor:

    def __init__(self, stop_words_directory, index_dir):
        self.stop_words_directory = stop_words_directory
        self.index_dir = index_dir
        
        stop_word_file = self.read_file(stop_words_directory)
        self.stop_words = word_tokenize(stop_word_file)
        
        with shelve.open(index_dir) as db:
            self.vocab_index = db['vocab_index']
            self.tfidf_matrix = db['tfidf_matrix']
            self.df = db['doc_freq']

        print("TF-IDF matrix, df_index, and vocab_index loaded successfully.")
        print(f"Vocabulary size: {len(self.vocab_index)}")
        print(f"Document Frequency Index: {len(self.df)}")
        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")

        self.V = len(self.vocab_index)
        self.N = len(self.tfidf_matrix)

        self.lemmatizer = WordNetLemmatizer()

    def read_file(self, filepath):
        return Path(filepath).read_text()
    
    def preprocess_and_get_features(self, filepath=None, stop_words=None, text=None):
        try:
            if text is None and filepath:
                text = self.read_file(filepath)
            elif text is None:
                raise ValueError("Either text or filepath must be provided.")

            tokens = []
            for word in word_tokenize(text):
                if re.match(r'\w+', word):
                    if '-' in word:
                        tokens.extend(word.split('-'))
                    else:
                        tokens.append(word)
                elif re.match(r'^\d{1,3}(?:,\d{3})*$', word):
                    tokens.append(word.replace(',', ''))

            stop_words = stop_words or self.stop_words

            return [
                self.lemmatizer.lemmatize(word.lower())
                for word in tokens
                if word.lower() not in stop_words
            ]
        except:
            return "Error In: Feature Extraction"

    def process_query(self, query):
        try:
            tokens = self.preprocess_and_get_features(text=query)
            tf = Counter(tokens)
            vec = np.zeros(self.V)
            
            for term, freq in tf.items():
                if term in self.vocab_index:
                    tf_val = 1 + math.log10(freq) if freq > 0 else 0
                    df = self.df[term]
                    idf_val = math.log10(self.N / df)
                    vec[self.vocab_index[term]] = tf_val * idf_val
            
            return vec.reshape(1, -1)
        except:
            return "Error In: Processing Query"

    def rank_documents(self, query, alpha=0.001):
        try:
            query_vector = self.process_query(query)
            query_vector = query_vector.flatten()  # Ensure it's 1D

            similarities = []
            for doc_vector in self.tfidf_matrix:
                score = 1 - cosine(query_vector, doc_vector)
                similarities.append(score)

            doc_scores = [(index + 1, score) for index, score in enumerate(similarities) if score > alpha]
            sorted_doc_scores = sorted(doc_scores, key=lambda x: x[1], reverse=True)
            ranked_doc_ids = [doc_id for doc_id, _ in sorted_doc_scores]
            ranked_scores = [score for _, score in sorted_doc_scores]

            return ranked_doc_ids, ranked_scores
        except:
            return "Error In: Ranking documents"
