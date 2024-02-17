# Restaurant-Chatbot

This simple restaurant chatbot is built using Flask and TensorFlow and can answer basic queries. 

## Deep Learning
The deep learning component of the chatbot involves using a neural network to understand and generate responses. Here's an overview of the key components:

### Neural Network Architecture

The model is built using TensorFlow and Keras. It consists of a feedforward neural network with the following layers:

- Input Layer: Dense layer with ReLU activation, taking input features from the bag-of-words representation.
- Dropout Layer: Helps prevent overfitting during training.
- Hidden Layer: Another Dense layer with ReLU activation.
- Output Layer: Dense layer with softmax activation for multi-class classification.

### Training

The model is trained using a dataset of intents and patterns. The patterns are tokenized, lemmatized, and converted into a bag-of-words representation. The output is a one-hot encoded vector indicating the predicted intent.

### Natural Language Processing (NLP)

The nltk library is used for natural language processing tasks, such as tokenization and lemmatization, to preprocess the input text.

## Future Improvements 
The intents and patterns can be expanded for a more diverse user experience.

## Screenshots
![Chatbot Interface](/screenshots/chatbot_interface.png)
