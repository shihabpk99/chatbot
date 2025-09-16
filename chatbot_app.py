import pickle
import numpy as np
import random
import json
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.svm import SVC
import re

# Initialize the stemmer
stemmer = PorterStemmer()

# --- Load the Model and Data ---
with open("chatbot_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("words.pkl", "rb") as words_file:
    words = pickle.load(words_file)
    
with open("labels.pkl", "rb") as labels_file:
    labels = pickle.load(labels_file)

with open("intents.json") as intents_file:
    intents = json.load(intents_file)

# A variable to store the user's name
user_name = None

def get_response(user_input):
    global user_name
    
    # Pre-process user input for prediction
    user_input_words = nltk.word_tokenize(user_input)
    user_input_words = [stemmer.stem(w.lower()) for w in user_input_words]
    
    bag_of_words = [0] * len(words)
    for w in user_input_words:
        if w in words:
            bag_of_words[words.index(w)] = 1
            
    # Predict the intent
    prediction = model.predict([bag_of_words])[0]
    
    # Check if the intent is for getting the user's name
    if prediction == 'get_name':
        name_match = re.search(r'(my name is|i am|you can call me|i\'m called|i go by)\s+([a-zA-Z]+)', user_input.lower())
        if name_match:
            user_name = name_match.group(2).capitalize()
            return f"Nice to meet you, {user_name}!"

    for intent in intents['intents']:
        if intent['tag'] == prediction:
            response = random.choice(intent['responses'])
            if user_name and intent['tag'] == 'greeting':
                 return f"{response} {user_name}!"
            return response
            
    # Fallback
    return random.choice(intents['intents'][5]['responses'])
    

def chat():
    print("Bot: Hi! I'm a machine-learning powered chatbot. Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("Bot: Goodbye!")
            break
        
        response = get_response(user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    chat()