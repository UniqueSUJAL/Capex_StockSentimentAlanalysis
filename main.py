import praw
import requests
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize nltk downloads
nltk.download('stopwords')
nltk.download('wordnet')

# Constants
NEWS_API_KEY = "aec04775-e3a7-4531-834c-fbd83d6964bb"  # Replace with your NewsAPI key
REDDIT_CLIENT_ID = "yCvEpcXTnyWGEYC5gA5raA"
REDDIT_CLIENT_SECRET = "RRqQYzIy96KMNTWiG07ECTX-Nxi-PQ"
REDDIT_USER_AGENT = "StockSentimentAnalysis"

# Initialize Reddit API
reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID, client_secret=REDDIT_CLIENT_SECRET, user_agent=REDDIT_USER_AGENT)

# Preprocessing utilities
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = emoji.demojize(text)  # Convert emojis to text
    text = ''.join([char.lower() if char.isalnum() or char.isspace() else '' for char in text])
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

# Fetch data from NewsAPI
def fetch_news_articles(keyword, max_results=10):
    url = f"https://newsapi.org/v2/everything?q={keyword}&apiKey={NEWS_API_KEY}&pageSize={max_results}"
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json().get('articles', [])
        return pd.DataFrame([{'Text': article['title'], 'Source': 'News', 'Date': article['publishedAt']} for article in articles])
    else:
        print(f"Failed to fetch news: {response.status_code}")
        return pd.DataFrame()

# Fetch data from Reddit
def fetch_reddit_posts(keyword, max_results=10):
    posts = reddit.subreddit('all').search(keyword, limit=max_results)
    return pd.DataFrame([{'Text': post.title, 'Source': 'Reddit', 'Date': pd.to_datetime(post.created_utc, unit='s')} for post in posts])

# Sentiment analysis using transformers
def analyze_sentiment(df):
    sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    df['Sentiment'] = df['Text'].apply(lambda x: sentiment_model(x[:512])[0]['label'])
    return df

# Machine learning-based sentiment analysis
def train_ml_sentiment_model(df):
    df['Target'] = df['Sentiment'].map({'POSITIVE': 1, 'NEGATIVE': 0})
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['Text'])
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print("Classification Report:")
    print(classification_report(y_test, predictions))
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return model, vectorizer, accuracy

# Visualize sentiment distribution
def visualize_sentiment(df):
    sentiment_counts = df['Sentiment'].value_counts()
    sentiment_counts.plot(kind='bar', color=['green', 'red'])
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Frequency")
    plt.show()

# SWOT Analysis
def swot_analysis(df):
    positives = len(df[df['Sentiment'] == 'POSITIVE'])
    negatives = len(df[df['Sentiment'] == 'NEGATIVE'])
    total = len(df)

    if positives / total > 0.6:
        strengths = "Strong positive sentiment. Good public opinion."
        opportunities = "Potential for investment or growth."
    else:
        strengths = "Moderate public support."
        opportunities = "Room for improvement in public perception."

    if negatives / total > 0.4:
        weaknesses = "Negative sentiment could impact brand reputation."
        threats = "Risk of declining trust or market position."
    else:
        weaknesses = "Limited negative feedback."
        threats = "Minimal risk at this time."

    return {"Strengths": strengths, "Weaknesses": weaknesses, "Opportunities": opportunities, "Threats": threats}

# Main script
if __name__ == "__main__":
    keyword = input("Enter the stock keyword: ")
    max_results = int(input("Enter the number of articles/posts to fetch: "))

    # Fetch data
    news_df = fetch_news_articles(keyword, max_results)
    reddit_df = fetch_reddit_posts(keyword, max_results)
    combined_df = pd.concat([news_df, reddit_df], ignore_index=True)

    # Preprocess and analyze
    combined_df['Text'] = combined_df['Text'].apply(clean_text)
    combined_df = analyze_sentiment(combined_df)

    # Visualize sentiment
    visualize_sentiment(combined_df)

    # Train ML model and get accuracy
    print("\nTraining Machine Learning Sentiment Model:")
    ml_model, vectorizer, accuracy = train_ml_sentiment_model(combined_df)
    print(f"ML Model Accuracy: {accuracy * 100:.2f}%")

    # Perform SWOT analysis
    swot = swot_analysis(combined_df)
    for key, value in swot.items():
        print(f"{key}: {value}")
