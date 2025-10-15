# 🍳 Recipe Recommendation System

An intelligent AI-powered recipe recommendation system that suggests recipes based on the ingredients you have at hand. Using advanced Natural Language Processing (NLP) and Machine Learning techniques, this system finds the most relevant recipes from a comprehensive database.

## 📹 Demo

<div align="center">
  
  https://github.com/sharathkum05/recipie-recommendation-system-main/assets/journal.mp4
  
  *🎥 Watch the Recipe Recommendation System in action!*
  
  > **Note:** If the video doesn't play above, you can [download and view the demo video here](./journal.mp4)
  
</div>

## ✨ Features

- 🔍 **Intelligent Recipe Search** - Find recipes by simply entering available ingredients
- 🤖 **NLP-Powered Matching** - Uses TF-IDF vectorization and cosine similarity for accurate recommendations
- 📊 **Comprehensive Recipe Details** - Get cuisine type, course, diet information, cooking time, and step-by-step instructions
- ⚡ **Fast & Efficient** - Pre-computed embeddings for quick recipe matching
- 🎨 **Beautiful UI** - Clean, modern, and responsive web interface
- 🕒 **Time-Based Filtering** - Recipes categorized by preparation time (under 10 mins, 30 mins, 1 hour, etc.)

## 🛠️ Tech Stack

### Backend
- **Flask** - Lightweight Python web framework
- **Python 3.x** - Core programming language

### Machine Learning & NLP
- **scikit-learn** - TF-IDF vectorization and cosine similarity
- **NLTK** - Natural Language Processing toolkit for text preprocessing
- **pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing

### Data Processing
- **WordNet Lemmatizer** - Text normalization
- **Porter Stemmer** - Word stemming
- **TF-IDF Vectorizer** - Feature extraction from text data

### Deployment
- **Gunicorn** - WSGI HTTP Server for production deployment
- **Dill** - Serialization for machine learning models

### Visualization & Analytics
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical data visualization
- **Plotly** - Interactive graphs

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd recipie-recommendation-system-main
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data**
   ```python
   python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
   ```

### Running the Application

1. **Start the Flask server**
   ```bash
   python app.py
   ```

2. **Access the application**
   
   Open your browser and navigate to: `http://localhost:5000`

3. **Enter ingredients**
   
   Type in the ingredients you have (e.g., "chicken, garlic, tomatoes") and click "Find Recipes"

## 📖 How It Works

### 1. **Data Preprocessing**
   - Cleans and normalizes recipe data from `Food_Recipe.csv`
   - Removes duplicates and null values
   - Standardizes text to title case
   - Calculates total cooking time and categorizes recipes

### 2. **NLP Pipeline**
   - **Tokenization**: Breaks down text into individual words
   - **Lemmatization**: Converts words to their base form using WordNet Lemmatizer
   - **Stop Words Removal**: Filters out common English words
   - **Punctuation Handling**: Removes special characters while preserving commas

### 3. **Feature Extraction**
   - Uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert recipe text into numerical vectors
   - Combines multiple features (ingredients, cuisine, course, diet, instructions) into unified embeddings

### 4. **Similarity Matching**
   - Applies **Cosine Similarity** to find recipes most similar to user input
   - Returns top N recipes ranked by similarity score
   - Provides detailed information for each recommended recipe

## 📁 Project Structure

```
recipie-recommendation-system-main/
│
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── Procfile                        # Deployment configuration
├── Food_Recipe.csv                 # Main recipe dataset
├── journal.mp4                     # Demo video
│
├── model/
│   ├── __init__.py                # Package initializer
│   ├── nlp_model.py               # NLP model and recommendation logic
│   ├── Food_Recipe.csv            # Recipe data for model
│   ├── combined_embeddings.pkl    # Pre-computed recipe embeddings
│   └── tfidf_vectorizer (1).pkl   # Trained TF-IDF vectorizer
│
├── static/
│   ├── styles.css                 # CSS styling
│   └── recipe*.jpg                # Recipe images
│
└── templates/
    ├── index.html                 # Home page
    └── results.html               # Results display page
```

## 🎯 Key Algorithms

### TF-IDF (Term Frequency-Inverse Document Frequency)
Converts recipe text into numerical vectors by weighing terms based on:
- **Term Frequency**: How often a word appears in a recipe
- **Inverse Document Frequency**: How unique the word is across all recipes

### Cosine Similarity
Measures the similarity between user input and recipe vectors:
```
similarity = (A · B) / (||A|| × ||B||)
```
Where A is the user input vector and B is a recipe vector.

## 📊 Dataset

The system uses a comprehensive recipe dataset (`Food_Recipe.csv`) containing:
- Recipe names
- Ingredients
- Cuisine types
- Course information (appetizer, main course, dessert, etc.)
- Dietary information (vegetarian, vegan, etc.)
- Cooking instructions
- Preparation and cooking times

## 🌐 Deployment

The application is configured for deployment with:
- **Procfile** for Heroku/similar platforms
- **Gunicorn** as the production WSGI server

To deploy:
```bash
gunicorn app:app
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available for educational and personal use.

## 🙏 Acknowledgments

- Recipe dataset contributors
- NLTK and scikit-learn communities
- Flask framework developers

---

**Made with ❤️ and Python**

