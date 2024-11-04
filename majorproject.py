# %%

import pandas as pd

# %%
df = pd.read_csv("C:\\Users\\dhuli\\Downloads\\IMDB Dataset.csv")

# %%
df

# %%
df['sentiment'].unique()

# %%
df['sentiment'][7]

# %%
df['review'][3]

# %%
import re

# %%
df['clean_text'] = df['review'].apply(lambda x: re.sub("<.*?>","",x))

# %%
df['clean_text'][3]

# %%
df['clean_text'] = df['clean_text'].apply(lambda x:re.sub(r'[^\w\s]',"",x))

# %%
df['clean_text'][3]

# %%
df['clean_text'] = df['clean_text'].str.lower()

# %%
df['clean_text'][3]

# %%
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

# %%
df['tokenize_text'] = df['clean_text'].apply(lambda x: word_tokenize(x))

# %%
df['tokenize_text'][1]

# %%
len(df['tokenize_text'][1])

# %%
nltk.download('stopwords')
from nltk.corpus import stopwords

# %%
stop_words = set(stopwords.words('english'))

# %%
stop_words

# %%
df['filter_text'] = df['tokenize_text'].apply(lambda x:[word for word in x if word not in stop_words])

# %%
len(df['filter_text'][1])

# %%
from nltk.stem import PorterStemmer

# %%
stem = PorterStemmer()

# %%
df['stem_text'] = df['filter_text'].apply(lambda x:[stem.stem(word) for word in x])

# %%
df['stem_text'][1]

# %%
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

# %%
lemma = WordNetLemmatizer()

# %%
df['lemma_text'] = df['filter_text'].apply(lambda x:[lemma.lemmatize(word) for word in x])

# %%
df['lemma_text'][1]

# %%
x=df['stem_text']
y=df['sentiment']

# %%
x

# %%
y

# %%
from sklearn.model_selection import train_test_split

# %%
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42,test_size=0.2)

# %%
x_train

# %%
x_test

# %%
y_test

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

# %%
tfidf = TfidfVectorizer()

# %%
x_train = tfidf.fit_transform(x_train.apply(lambda x:''.join(x)))

# %%
x_test = tfidf.transform(x_test.apply(lambda x:"".join(x)))

# %%
from sklearn.preprocessing import LabelEncoder

# %%
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# %%
from keras.utils import to_categorical

# %%
y_train = to_categorical(y_train,num_classes =2)

# %%
y_train

# %%
x_test.shape

# %%
type(x_train)

# %%
from keras import Sequential

# %%
from keras.layers import Dense

# %%
model = Sequential([
    Dense(128,activation="relu",input_shape=(x_train.shape[1],)),
    Dense(64,activation="relu"),
    Dense(32,activation="relu"),
    Dense(2,activation="sigmoid")])

# %%
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])

# %%
model.fit(x_train,y_train,epochs=10)

# %%
import streamlit as st
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

#joblib.dump(model,'model.pkl')
#joblib.dump(tfidf,'tfidf.pkl')

model_loaded=joblib.load('model.pkl')
tf_idf_vector=joblib.load('tfidf.pkl')

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def predict_sentiment(review):
    cleaned_review = re.sub('<.*?>','',review)
    cleaned_review = re.sub(r'[^\w\s]','',cleaned_review)
    cleaned_review = cleaned_review.lower()
    tokenized_review = word_tokenize(cleaned_review)
    filtered_review = [word for word in tokenized_review if word not in stop_words]
    stemmed_review = [stemmer.stem(word) for word in filtered_review]
    tfidf_review = tf_idf_vector.transform([' '.join(stemmed_review)])
    sentiment_prediction = model.predict(tfidf_review)
    if sentiment_prediction> 0.6:
        return "Positive"
    else:
        return "Negative"

st.title('SENTIMENT ANALYSIS')
review_to_predict = st.text_area('Enter your review here:')
if st.button('Predict Sentiment'):
    predicted_sentiment = predict_sentiment(review_to_predict)
    st.write("Predicted Sentiment:" ,predicted_sentiment)

