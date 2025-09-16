import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np
import random
import json
import pickle
from sklearn.svm import SVC

# Initialize the stemmer
stemmer = PorterStemmer()

# --- Load and Pre-process Data ---
with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '!', '.', ',']
all_words = [stemmer.stem(w.lower()) for w in all_words if w not in ignore_words]
all_words = sorted(list(set(all_words)))
tags = sorted(list(set(tags)))

# --- Create Training Data ---
training_data = []
all_labels = []

for (pattern_sentence, tag) in xy:
    bag = [0] * len(all_words)
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_sentence]
    for idx, w in enumerate(all_words):
        if w in pattern_words:
            bag[idx] = 1
    
    training_data.append(bag)
    all_labels.append(tag)

X_train = np.array(training_data)
y_train = np.array(all_labels)

# --- Train the Model ---
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# --- Save the Model and Data ---
with open("chatbot_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("words.pkl", "wb") as f:
    pickle.dump(all_words, f)
    
with open("labels.pkl", "wb") as f:
    pickle.dump(tags, f)

print("Training complete. Model and data saved to files.")