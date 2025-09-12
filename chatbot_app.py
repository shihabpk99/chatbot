import pickle
import numpy as np
import random
import json
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Initialize the stemmer
stemmer = PorterStemmer()

# Define the custom tokenizer function (The Fix!)
def custom_tokenizer(text):
    """Tokenizes and stems the input text."""
    words = nltk.word_tokenize(text)
    stemmed_words = [stemmer.stem(w.lower()) for w in words]
    return stemmed_words

# Load the trained pipeline (which includes the vectorizer and classifier)
with open("chatbot_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)
with open("intents.json") as intents_file:
    intents = json.load(intents_file)

def get_response(user_input):
    """
    Analyzes user input using the trained model and provides a response.
    """
    # Use the loaded model's predict function
    # It will automatically use the custom tokenizer within the pipeline
    prediction = model.predict([user_input])[0]

    # Find the intent with the highest probability
    for intent in intents['intents']:
        if intent['tag'] == prediction:
            return random.choice(intent['responses'])
    
    # Fallback response if no intent is found (this is unlikely with this model type)
    return random.choice(intents['intents'][4]['responses']) # Assuming 'no_answer' is the 5th intent

# Main chat loop
def chat():
    print("Bot: Hi! I'm a machine-learning powered chatbot. Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("Bot: Goodbye!")
            break
        
        response = get_response(user_input)
        print(f"Bot: {response}")

# Start the chatbot
if __name__ == "__main__":
    chat()