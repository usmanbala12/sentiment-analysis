# Twitter Sentiment Analysis

A machine learning project for classifying Twitter tweets as positive or negative sentiment using the Sentiment140 dataset.

## Overview

This project trains a logistic regression model to perform binary sentiment classification on Sentiment140 dataset. The dataset contains 160,000 tweets with balanced positive and negative sentiment labels.

## Project Structure

- `Sentiment.ipynb` - Main Jupyter notebook containing the analysis
- `MP2_Sentiment140.tsv` - Dataset file (Sentiment140)

## Workflow

1. **Data Processing and EDA** - Load, clean, and explore the dataset
2. **Feature Selection and Engineering** - Text preprocessing including tokenization, stemming, and TF-IDF vectorization
3. **Model Training and Evaluation** - Train Logistics Regression model and evaluate performance

## Requirements

- Python 3.9.25
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- NLTK
- TensorFlow/Keras
- BeautifulSoup4
- WordCloud

## Usage

1. Install dependencies:

   ```
   pip install beautifulsoup4 seaborn matplotlib wordcloud scikit-learn keras nltk tensorflow
   ```

2. Open and run the Jupyter notebook:
   ```
   jupyter notebook Sentiment.ipynb
   ```

## Author

Usman Bala Usman (contact.usmanusman@gmail.com)
