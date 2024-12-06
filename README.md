# Stock Sentiment Analysis

## Project Overview
The **Stock Sentiment Analysis** project helps analyze public sentiment about specific stocks by extracting data from **News articles** and **Reddit posts**. This project combines **Natural Language Processing (NLP)** and **Machine Learning (ML)** techniques to determine sentiment and provide insights like **SWOT analysis** (Strengths, Weaknesses, Opportunities, Threats).

It uses data from **Reddit** (via the **Praw** library) and **NewsAPI** (to fetch news articles). The sentiment analysis is performed using both a **pre-trained transformer model** (`DistilBERT`) and a **Naive Bayes** machine learning model.

## Key Features
- **Data Collection**: Fetches the latest news articles and Reddit posts about a specific stock.
- **Sentiment Analysis**: 
  - Uses a **pre-trained sentiment analysis model** (`DistilBERT`) from **Hugging Face Transformers**.
  - Trains a **Naive Bayes** machine learning model to predict sentiment.
- **Text Preprocessing**: Cleans and processes the text data by removing stopwords, lemmatizing words, and converting emojis to readable text.
- **SWOT Analysis**: Provides insights based on sentiment to evaluate the **Strengths**, **Weaknesses**, **Opportunities**, and **Threats** related to the stock.
- **Data Visualization**: Displays sentiment distribution (Positive vs Negative) in a **bar chart** using **Matplotlib**.

## Installation

### Prerequisites

Before starting, ensure that you have **Python 3.x** installed. If you don’t have Python installed, you can download it from [here](https://www.python.org/downloads/).

You’ll also need some Python packages to run this project. These packages are used for fetching the data, performing sentiment analysis, and visualizing results.

Here are the steps to set it up:

1. **Install Python Packages**
   
   You’ll need to install the required Python libraries. To make it easy, this project uses a `requirements.txt` file which lists all the packages you need. To install them, follow these steps:
   
   - Open a **terminal** or **command prompt** (depending on your operating system).
   - Navigate to the project folder where this README file is located. For example:
     ```bash
     cd path_to_your_project_folder
     ```
   - Run the following command to install all dependencies:
     ```bash
     pip install -r requirements.txt
     ```
     This will automatically install all the necessary packages for you. If you don’t have `pip` (Python’s package installer), you can install it from [here](https://pip.pypa.io/en/stable/installation/).

   The key packages you'll need are:
   - `praw`: For interacting with Reddit API and fetching posts.
   - `requests`: For fetching news articles from NewsAPI.
   - `pandas`: For data manipulation.
   - `transformers`: For using pre-trained models for sentiment analysis.
   - `matplotlib`: For plotting graphs (e.g., sentiment distribution).
   - `scikit-learn`: For the machine learning-based sentiment analysis.
   - `emoji`: To handle emojis in text data.
   - `nltk`: For text preprocessing (like removing stopwords).

2. **Get API Keys**
   To fetch data from **NewsAPI** and **Reddit**, you need API keys:
   
   - **NewsAPI**:
     - Go to [NewsAPI](https://newsapi.org/) and create an account to get your API key.
     - Once you have your key, replace the placeholder in the script `sentiment_analysis.py` with your NewsAPI key.
   
   - **Reddit API**:
     - Create a Reddit application by visiting [Reddit App](https://www.reddit.com/prefs/apps).
     - Get your **client ID**, **client secret**, and **user agent**.
     - Replace the placeholders in the script `sentiment_analysis.py` with your Reddit credentials.

3. **Set Up NLTK (Natural Language Toolkit)**
   The project uses **NLTK** to process text (removing stopwords, lemmatizing, etc.). NLTK needs to download a couple of datasets to work properly. Run the following in Python to ensure the necessary files are downloaded:
   
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
### Installing
4. **Clone the repository** to your local machine:
   ```bash
   git clone https://github.com/UniqueSUJAL/Capex_StockSentimentAnalysis.git
