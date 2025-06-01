from flask import Flask, render_template, request
#from model import recipe_logic
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from model import nlp_model
import pickle, string
import nltk
from nltk.corpus import stopwords
import numpy as np

app = Flask(__name__)

data = pd.read_csv("./model/Food_Recipe.csv")

new_df = data[['name', 'description', 'cuisine','course','diet', 'ingredients_name', 'prep_time (in mins)','cook_time (in mins)', 'instructions']]

cat_cols = (new_df.dtypes[new_df.dtypes == 'object']).index

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


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/recommend", methods=["POST"])
def recommend():
    text = request.form.get("ingredients", "")
    text = str(text)

    recipes = nlp_model.find_similar_recipes(new_df, text, 6)
    selected_recipes = new_df.loc[new_df['name'].isin(recipes), ['name', 'ingredients_name', 'cuisine', 'course', 'diet', 'instructions', 'total_time']]
    selected_recipes.reset_index(drop=True, inplace=True)

    return render_template("results.html", recipes=selected_recipes.to_dict(orient="records"))

if __name__ == "__main__":
    app.run(debug=True)
