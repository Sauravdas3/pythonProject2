# importing packages
from fastapi import FastAPI
import uvicorn
import numpy as np
import pickle
import pandas as pd
import string

import nltk
from nltk.stem import WordNetLemmatizer
import contractions
from nltk.corpus import stopwords
from fastapi.responses import HTMLResponse
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000)



nltk.download('wordnet')  # Download WordNet data
wnl = WordNetLemmatizer()

w = []
c = []
def create_new_features(text):
    words = text.split()
    char_len = 0
    for word in words:
        char_len += len(word)
    w.append(len(words))
    c.append(char_len)
    return (len(words), char_len)


nltk.download('stopwords')
import re
def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F" # emoticons
                           u"\U0001F300-\U0001F5FF" # symbols & pictographs
                           u"\U0001F680-\U0001F6FF" # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF" # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')  # Download WordNet data
wnl = WordNetLemmatizer()
def preprocess_text(text):
    #remove html tags
    text = re.compile(r'<[^>]+>').sub('', text)
    # convert to lower case
    text = text.lower()
    # expand contractions
    text = contractions.fix(text)
    # remove punctuations
    text = text.translate(str.maketrans('','', string.punctuation))
    # remove numbers
    text = ''.join([i for i in text if not i.isdigit()])
    # remove stop words
    text = ' '.join([word for word in text.split() if word not in (stopwords.words('english'))])
    # perform stemming
    text = text.split()
    words = []
    for word in text:
        words.append(wnl.lemmatize(word))
    text = ' '.join(words)
    # remove emoji
    text = remove_emoji(text)
    # remove extra spaces
    text = ' '.join(text.split())
    return text




# creating api name
app = FastAPI()
pickle_in = open("model.pkl","rb")
classifier = pickle.load(pickle_in)


# Index route, opens automatically on http//127.0.0.1:1111
@app.get("/", response_class=HTMLResponse)
def get_message():
    original_message = "Welcome To my 2nd project\nI am Saurav, the creator of the project\nmy mail id: sauravdasdas07@gmail.com"
    lines = original_message.split("\n")
    formatted_message = "<br>".join(lines)
    return formatted_message

@app.post('/predict')
def predict_emotion(text: str):
    result = ''
    try:

        text = preprocess_text(text)
        word_len, char_len = create_new_features(text)
        text_features = cv.transform([text])
        text_features_dense = text_features.toarray()
        zero_arr = np.array([0])
        one_arr = np.array([1])
        char_len_arr = np.array([char_len])
        word_len_arr = np.array([word_len])
        if text_features_dense.ndim > 1:
            text_features_dense = np.squeeze(text_features_dense)
        features = np.hstack((text_features_dense, zero_arr, one_arr, char_len_arr, word_len_arr))
        features = features.reshape(1, -1)

        # Predict sentiment
        prediction = classifier.predict(features)

        if prediction < 5:
            result = "not satisfied"
        elif prediction == 5:
            result = "Neutral"
        else:
            result = "Satisfied"
        return {
            'prediction': result
        }
    except Exception as e:
        return {"error": str(e)}




if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=1111)





