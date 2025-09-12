import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

# Initialize the stemmer
stemmer = PorterStemmer()

# Define the custom tokenizer function (The Fix!)
def custom_tokenizer(text):
    """Tokenizes and stems the input text."""
    words = nltk.word_tokenize(text)
    stemmed_words = [stemmer.stem(w.lower()) for w in words]
    return stemmed_words

# --- Step 1: Load Data ---
with open('intents.json', 'r') as f:
    intents = json.load(f)

# --- Step 2: Prepare the Training Data ---
# Create lists to hold the training sentences and their corresponding labels (tags)
all_training_sentences = []
all_labels = []

for intent in intents['intents']:
    tag = intent['tag']
    # Add all patterns for this intent to our training lists
    for pattern in intent['patterns']:
        all_training_sentences.append(pattern)
        all_labels.append(tag)

# --- Step 3: Train the Model ---
# Create a pipeline with a TF-IDF vectorizer and an SVM classifier
# We pass our named tokenizer function to the TfidfVectorizer
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(tokenizer=custom_tokenizer, lowercase=False)),
    ('classifier', SVC(kernel='linear'))
])

# Train the pipeline
pipeline.fit(all_training_sentences, all_labels)

# --- Step 4: Save the Model and Data ---
# Save the trained pipeline to a file
with open("chatbot_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Training complete. Model and data saved to chatbot_model.pkl.")