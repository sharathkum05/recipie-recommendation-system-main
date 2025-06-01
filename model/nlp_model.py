# Import libraries
# import gensim
# from gensim.models import Word2Vec
import pickle, string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from collections import Counter
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
import warnings
warnings.filterwarnings('ignore')
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("stopwords")
nltk.download("wordnet")

df = pd.read_csv("./Food_Recipe.csv")
#Drop uneccsary columns for model
new_df = df[['name', 'description', 'cuisine','course','diet', 'ingredients_name', 'prep_time (in mins)','cook_time (in mins)', 'instructions']]

#Define all categorical columns
cat_cols = (new_df.dtypes[new_df.dtypes == 'object']).index

#Convert all object columns to title case
for col in cat_cols:
    new_df[col] = new_df[col].str.title()


new_df.dropna(inplace=True)

new_df.drop_duplicates(inplace=True)

new_df['total_time'] = np.where(new_df['prep_time (in mins)'] + new_df['cook_time (in mins)'] == 0, 'no prep',
                            np.where(new_df['prep_time (in mins)'] + new_df['cook_time (in mins)'] <= 10, 'under 10 mins',
                            np.where(new_df['prep_time (in mins)'] + new_df['cook_time (in mins)'] <= 30, 'under 30 mins',
                                     np.where(new_df['prep_time (in mins)'] + new_df['cook_time (in mins)'] <= 60, 'under 1 hour',
                                              np.where(new_df['prep_time (in mins)'] + new_df['cook_time (in mins)'] <= 300, 'under 5 hours',
                                                       np.where(new_df['prep_time (in mins)'] + new_df['cook_time (in mins)'] > 300, 'more than 5 hours',
                                                                'na'))))))

new_df.drop(['prep_time (in mins)', 'cook_time (in mins)'], axis=1, inplace=True)

new_df = new_df.sample(frac=1, random_state=123)

stemmer = nltk.stem.PorterStemmer()
ENGLISH_STOP_WORDS = stopwords.words('english')

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def recipe_tokenizer(sentence):
    sentence = ''.join([char if char not in string.punctuation or char == ',' else ' ' for char in sentence]).lower()
    listofwords = sentence.split()
    listoflemmatized_words = [lemmatizer.lemmatize(word) for word in listofwords if word not in ENGLISH_STOP_WORDS and word != '']

    return listoflemmatized_words

def load_embeddings_and_vectorizer():
    with open("model/combined_embeddings.pkl", 'rb') as f:
        combined_embeddings = pickle.load(f)
    with open('model/tfidf_vectorizer (1).pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return combined_embeddings, vectorizer

def find_similar_recipes(sampled_data, user_input, num_similar=5):
    try:
        combined_embeddings, vectorizer = load_embeddings_and_vectorizer()
    except FileNotFoundError:
        precompute_embeddings(sampled_data)
        combined_embeddings, vectorizer = load_embeddings_and_vectorizer()

    user_data = pd.DataFrame({'text_data': [user_input]})
    user_data['text_data'] = user_data['text_data'].str.lower()

    user_vectorized_data = vectorizer.transform(user_data['text_data'])

    num_missing_features = combined_embeddings.shape[1] - user_vectorized_data.shape[1]
    if num_missing_features > 0:
        user_vectorized_data = np.pad(user_vectorized_data.toarray(), ((0, 0), (0, num_missing_features)))

    cosine_sim_matrix = cosine_similarity(user_vectorized_data, combined_embeddings)

    similar_recipes = cosine_sim_matrix[0].argsort()[::-1][:num_similar]
    similarity_scores = cosine_sim_matrix[0][similar_recipes]

    similar_recipe_names = sampled_data.iloc[similar_recipes]['name'].tolist()

    similar_recipes_with_scores = list(zip(similar_recipe_names, similarity_scores))

    recipies = []
    for recipe, score in similar_recipes_with_scores:
        recipies.append(recipe)
        print(f"Recipe: {recipe}, Similarity Score: {score:.4f}")

    return recipies

