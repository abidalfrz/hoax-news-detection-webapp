import pandas as pd
import numpy as np
import pickle
import os
import emoji
import re
import unicodedata
import ftfy
from nltk.tokenize import word_tokenize, sent_tokenize
from textblob import TextBlob
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

SCALER_PATH = os.path.join(os.path.dirname(__file__), '../artifacts/scaler.pkl')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), '../artifacts/vectorizer.pkl')

def demojize_emoji(text):
    emoji_pattern = re.compile(
        "["
         u"\U0001F600-\U0001F64F"  # emoticons
         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
         u"\U0001F680-\U0001F6FF"  # transport & map symbols
         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
         u"\U00002702-\U000027B0"
         u"\U000024C2-\U0001F251"
         "]+", flags=re.UNICODE
       )
    
    if bool(emoji_pattern.search(text)):
        return re.sub("[\W_]"," ",emoji.demojize(text, delimiters=("", " ")))
    return text

def normalize_unicode(text):
    text = ftfy.fix_text(text)
    return "".join(
        c for c in unicodedata.normalize('NFKD', text)
        if not unicodedata.combining(c)
    )

def cleaned_text(text):

    text = text.lower()
    text = demojize_emoji(text)
    text = normalize_unicode(text)
    
    text = re.sub(r'(im)\s', 'i am ', text)
    text = re.sub(r"i[' ]m", "i am", text)
    text = re.sub(r"can[' ]t", "can not", text)
    text = re.sub(r"don[' ]t", "do not", text)
    text = re.sub(r"shouldn[' ]t", "should not", text)
    text = re.sub(r"needn[' ]t", "need not", text)
    text = re.sub(r"hasn[' ]t", "has not", text)
    text = re.sub(r"haven[' ]t", "have not", text)
    text = re.sub(r"weren[' ]t", "were not", text)
    text = re.sub(r"mightn[' ]t", "might not", text)
    text = re.sub(r"didn[' ]t", "did not", text)
    text = re.sub(r"it[' ]s ", "it is ", text)
    text = re.sub(r"(?:'| )re ", " are ", text)
    text = re.sub(r"(?:'| )s ", " is ", text)
    text = re.sub(r"(?:'| )d ", " would ", text)
    text = re.sub(r"(?:'| )ll ", " will ", text)
    text = re.sub(r"n(?:'| )t ", " not ", text)
    text = re.sub(r"(?:'| )ve ", " have ", text)
    text = re.sub(r"(?:'| )t ", " not ", text)
    text = re.sub("gonna|gon na", "going to", text)
    text = re.sub("wanna|wan na", "want to", text)
    text = re.sub("rn", "right now", text)
    text = re.sub("idk", 'i dont know', text)
    text = re.sub(r"http?://\S+|www\.\S+|https?://\S+", '', text)
    #text = re.sub('[^a-zA-Z0-9\s]', "", text)
    text = "".join([word for word in text if not word in string.punctuation])
    text = re.sub('[\n\t\r]', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip()

    return text
    
def add_features(df):
    df['word_count'] = df['text'].apply(lambda x: len(word_tokenize(x)))
    df['sentence_count'] = df['text'].apply(lambda x: len(sent_tokenize(x)))
    df['lexical_diversity'] = df['text'].apply(lambda x: len(set(word_tokenize(x)))) / df['word_count']
    df['polarity'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['subjectivity'] = df['text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

    return df

def preprocess(df, col='cleaned_text'):
    STOPWORDS= set(stopwords.words('english'))
    Lemma = WordNetLemmatizer()
    df[col] = df[col].apply(lambda x: " ".join([Lemma.lemmatize(word) for word in x.split() if word not in STOPWORDS]))
    return df

def preprocess_data(df):
    df['cleaned_text'] = df['text'].apply(cleaned_text)
    df = add_features(df)
    df = preprocess(df, col='cleaned_text')

    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    numeric = df.select_dtypes(include=['number']).columns.tolist()
    df[numeric] = scaler.transform(df[numeric])
    
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    
    text_vectors = vectorizer.transform(df['cleaned_text']).toarray()
    text_vector_df = pd.DataFrame(text_vectors, columns=vectorizer.get_feature_names_out())
    
    new_df = pd.concat([df[numeric].reset_index(drop=True), text_vector_df.reset_index(drop=True)], axis=1)
    return new_df
    