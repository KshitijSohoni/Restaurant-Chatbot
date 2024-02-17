from flask import Flask, render_template, request, jsonify
import json 
import string 
import random
import nltk 
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense, Dropout

app = Flask(__name__)

# Load NLTK data
nltk.download("punkt")
nltk.download("wordnet")

# Load intents data
file_path = 'intents_chatgpt.json'
with open(file_path, 'r') as file:
    data = json.load(file)

# Preprocess data
words = []
classes = []
data_x = []
data_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        data_x.append(pattern)
        data_y.append(intent["tag"])

        if intent["tag"] not in classes:
            classes.append(intent["tag"])

lemmatizer = WordNetLemmatizer()

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]
words = sorted(set(words))
classes = sorted(set(classes))

# Training data
training = []
out_empty = [0] * len(classes)

for idx, doc in enumerate(data_x):
    bow = []
    text = lemmatizer.lemmatize(doc.lower())
    for word in words:
        bow.append(1) if word in text else bow.append(0)

    output_row = list(out_empty)
    output_row[classes.index(data_y[idx])] = 1

    training.append([bow, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)
train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Load the pre-trained model
model = load_model("chatbot_model.h5")

# Functions (modify as needed)
def clean_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

def bag_of_words(text, vocab):
    tokens = clean_text(text)
    bow = [0] * len(vocab)
    for w in tokens:
        for idx, word in enumerate(vocab):
            if word == w:
                bow[idx] = 1
    return np.array(bow)

def pred_class(text, vocab, labels):
    bow = bag_of_words(text, vocab)
    result = model.predict(np.array([bow]))[0]
    thresh = 0.5
    y_pred = [[indx, res] for indx, res in enumerate(result) if res > thresh]
    y_pred.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in y_pred:
        return_list.append(labels[r[0]])
    return return_list

def get_response(intents_list, intents_json):
    if len(intents_list) == 0:
        result = "Sorry! I don't understand."
    else:
        tag = intents_list[0]
        list_of_intents = intents_json["intents"]
        for i in list_of_intents:
            if i["tag"] == tag:
                result = random.choice(i["responses"])
                break
    return result 

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_bot_response')
def get_bot_response():
    message = request.args.get('msg')
    app.logger.info(f"Received message: {message}")
    if message:
        intents = pred_class(message, words, classes)
        response = get_response(intents, data)
        app.logger.info(f"Generated response: {response}")
        return jsonify({'response': response})
    return jsonify({'error': 'No message provided'})


# Main
if __name__ == '__main__':
    app.run(debug=True)

