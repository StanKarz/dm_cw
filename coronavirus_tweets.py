# Part 3: Mining text data.
import numpy as np
import pandas as pd 
import requests
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Part 3: Mining text data.

# Return a pandas dataframe containing the data set.
# Specify a 'latin-1' encoding when reading the data.
# data_file will be populated with the string 'wholesale_customers.csv'.
def read_csv_3(data_file):
	df = pd.read_csv(data_file, encoding='latin-1')
	return df

# Return a list with the possible sentiments that a tweet might have.
def get_sentiments(df):
	return list(df['Sentiment'].unique())

# Return a string containing the second most popular sentiment among the tweets.
def second_most_popular_sentiment(df):
	return df['Sentiment'].value_counts().index[1]

# Return the date (string as it appears in the data) with the greatest number of extremely positive tweets.
def date_most_popular_tweets(df):
	return df[df['Sentiment'] == "Extremely Positive"].value_counts(subset=('TweetAt')).index[0]
	
# Modify the dataframe df by converting all tweets to lower case. 
def lower_case(df):
	df['OriginalTweet'] = df['OriginalTweet'].str.lower()
	return df

# Modify the dataframe df by replacing each characters which is not alphabetic or whitespace with a whitespace.
def remove_non_alphabetic_chars(df):
	# df.strings.str.replace('[^A-Za-z\s]', '')
	df['OriginalTweet'] = df['OriginalTweet'].replace('[^a-zA-Z\s]', ' ', regex=True)
	return df

# Modify the dataframe df with tweets after removing characters which are not alphabetic or whitespaces.
def remove_multiple_consecutive_whitespaces(df):
	df['OriginalTweet'] = df['OriginalTweet'].replace('\s+', ' ', regex=True).str.strip()
	return df
	

# Given a dataframe where each tweet is one string with words separated by single whitespaces,
# tokenize every tweet by converting it into a list of words (strings).
def tokenize(df):
	df['OriginalTweet'] = df['OriginalTweet'].str.split(' ')
	return df
	

# Given dataframe tdf with the tweets tokenized, return the number of words in all tweets including repetitions.
def count_words_with_repetitions(tdf):
	word_count = tdf['OriginalTweet'].apply(len).sum()
	return word_count

# Given dataframe tdf with the tweets tokenized, return the number of distinct words in all tweets.
def count_words_without_repetitions(tdf):
	unique_word_count = tdf.explode('OriginalTweet')['OriginalTweet'].nunique()
	return unique_word_count

# Given dataframe tdf with the tweets tokenized, return a list with the k distinct words that are most frequent in the tweets.
def frequent_words(tdf,k):
	k_distinct_words = tdf.explode("OriginalTweet")["OriginalTweet"].value_counts().head(k)
	return k_distinct_words.index.tolist()

# Given dataframe tdf with the tweets tokenized, remove stop words and words with <=2 characters from each tweet.
# The function should download the list of stop words via:
# https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt
def remove_stop_words(tdf):
	stop_words_url = "https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt"
	 # Download the stop words
	response = requests.get(stop_words_url)
	stop_words = set(response.text.splitlines())

    # Apply functon to remove stop words and words <= 2 characters
	tdf['OriginalTweet'] = tdf['OriginalTweet'].apply(lambda words: [word for word in words if word not in stop_words and len(word) > 2])
	return df

# Given dataframe tdf with the tweets tokenized, reduce each word in every tweet to its stem.
def stemming(tdf):
	stemmer = PorterStemmer()
	tdf['OriginalTweet'] = tdf['OriginalTweet'].apply(lambda words: [stemmer.stem(word) for word in words])
	return tdf

# Given a pandas dataframe df with the original coronavirus_tweets.csv data set,
# build a Multinomial Naive Bayes classifier. 
# Return predicted sentiments (e.g. 'Neutral', 'Positive') for the training set
# as a 1d array (numpy.ndarray). 
def mnb_predict(df):
	clf = MultinomialNB()
	vectorizer = CountVectorizer()
	X = vectorizer.fit_transform(df['OriginalTweet'].apply(lambda word: ' '.join(word)))
	y = df['Sentiment'].values
	clf.fit(X, y)
	return clf.predict(X)



# Given a 1d array (numpy.ndarray) y_pred with predicted labels (e.g. 'Neutral', 'Positive') 
# by a classifier and another 1d array y_true with the true labels, 
# return the classification accuracy rounded in the 3rd decimal digit.
def mnb_accuracy(y_pred,y_true):
	return accuracy_score(y_true, y_pred)








