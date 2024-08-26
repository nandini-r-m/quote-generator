from flask import Flask, send_from_directory, request
import pandas as pd
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import random
import json

app = Flask(__name__)

# Load the JSON file
json_file_path = 'AnimeQuotes.json'  # Update this with your actual file path
with open(json_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Convert the JSON data to a DataFrame
df = pd.DataFrame(data)

# Preprocess the text data
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

df['processed_quotes'] = df['Quote'].apply(preprocess_text)

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to get sentiment scores
def get_sentiment(text):
    return analyzer.polarity_scores(text)

# Apply sentiment analysis
df['sentiment'] = df['processed_quotes'].apply(get_sentiment)

# Extract compound score for simplicity
df['compound'] = df['sentiment'].apply(lambda x: x['compound'])

# Classify sentiment
def classify_sentiment(compound):
    if compound >= 0.05:
        return 'positive'
    elif compound <= -0.05:
        return 'negative'
    else:
        return 'neutral'

df['sentiment_class'] = df['compound'].apply(classify_sentiment)

# Function to get quotes based on mood
def get_quotes_by_mood(mood, num_quotes=1):  # Adjusted to return 1 quote by default
    if mood not in ['positive', 'negative', 'neutral']:
        raise ValueError("Mood must be 'positive', 'negative', or 'neutral'.")
    
    filtered_df = df[df['sentiment_class'] == mood]
    return filtered_df['Quote'].sample(n=num_quotes).tolist()


@app.route('/', methods=['GET', 'POST'])
def index():
    mood = None
    quotes = []
    if request.method == 'POST':
        mood = request.form.get('mood')
        quotes = get_quotes_by_mood(mood)
    
    return send_from_directory('', 'index.html'), {'mood': mood, 'quotes': quotes}

if __name__ == '__main__':
    app.run(debug=True)
