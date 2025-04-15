import math
import re
import numpy as np
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from pathlib import Path
import shelve


class VectorSpaceQueryProcessor:
    def __init__(self, stop_words_directory, index_dir):
        try:
            self.stop_words_directory = stop_words_directory
            self.index_dir = index_dir

            stop_word_file = self.read_file(stop_words_directory)
            if isinstance(stop_word_file, str) and stop_word_file.startswith("Error In"):
                raise Exception(stop_word_file)
            self.stop_words = word_tokenize(stop_word_file)

            with shelve.open(index_dir) as db:
                self.vocab_index = db['vocab_index']
                self.tfidf_matrix = db['tfidf_matrix']
                self.df = db['doc_freq']
                self.doc_norms = db['norms']

            print("TF-IDF matrix and vocab_index loaded successfully.")
            print(f"Vocabulary size: {len(self.vocab_index)}")
            print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")

            self.V = len(self.vocab_index)
            self.N = len(self.tfidf_matrix)
            self.lemmatizer = WordNetLemmatizer()

        except Exception as e:
            print(f"Error In: __init__ -> {str(e)}")

    def read_file(self, filepath):
        try:
            return Path(filepath).read_text()
        except Exception as e:
            return f"Error In: read_file -> {str(e)}"

    def preprocess_and_get_features(self, text):
        try:
            text = text.lower()
            text = re.sub(r'[^a-zA-Z\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            tokens = word_tokenize(text)
            clean_tokens = [
                self.lemmatizer.lemmatize(token)
                for token in tokens
                if token.isalpha() and token not in self.stop_words
            ]
            return clean_tokens
        except Exception as e:
            return f"Error In: preprocess_and_get_features -> {str(e)}"

    def process_query(self, query):
        try:
            tokens = self.preprocess_and_get_features(text=query)
            if isinstance(tokens, str):
                return tokens

            tf = Counter(tokens)
            vec = np.zeros(self.V)

            for term, freq in tf.items():
                if term in self.vocab_index:
                    tf_val = 1 + math.log10(freq) if freq > 0 else 0
                    df = self.df[term]
                    idf_val = math.log10(self.N / df)
                    vec[self.vocab_index[term]] = tf_val * idf_val

            return vec.reshape(1, -1)
        except Exception as e:
            return f"Error In: process_query -> {str(e)}"

    def rank_documents(self, query, alpha=0.001):
        try:
            tokens = self.preprocess_and_get_features(text=query)
            if isinstance(tokens, str):
                return tokens

            tf = Counter(tokens)
            query_values = {}
            query_norm_sq = 0.0

            for term, freq in tf.items():
                if term in self.vocab_index:
                    idx = self.vocab_index[term]
                    tf_val = 1 + math.log10(freq)
                    df = self.df[term]
                    idf_val = math.log10(self.N / df)
                    weight = tf_val * idf_val
                    query_values[idx] = weight
                    query_norm_sq += weight ** 2

            if not query_values or query_norm_sq == 0:
                return [], []

            query_norm = math.sqrt(query_norm_sq)
            scores = []

            for doc_id in range(self.N):
                dot_product = 0.0
                for idx, q_weight in query_values.items():
                    dot_product += q_weight * self.tfidf_matrix[doc_id][idx]

                doc_norm = self.doc_norms[doc_id]
                similarity = dot_product / (query_norm * doc_norm) if doc_norm != 0 else 0.0

                if similarity > alpha:
                    scores.append((doc_id + 1, round(similarity, 4)))

            sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
            ranked_doc_ids = [doc_id for doc_id, _ in sorted_scores]
            ranked_scores = [score for _, score in sorted_scores]

            return ranked_doc_ids[:100], ranked_scores[:100]
        except Exception as e:
            return f"Error In: rank_documents -> {str(e)}"
