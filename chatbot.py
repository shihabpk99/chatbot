import random

# A dictionary containing lists of keywords and corresponding responses
responses = {
    "hello": ["Hello there!", "Hi! How can I help you?", "Hey, it's great to see you!"],
    "how are you": ["I'm doing well, thank you for asking!", "As a bot, I'm always ready to assist.", "I'm functioning perfectly!"],
    "bye": ["Goodbye!", "See you later!", "Talk to you soon!"],
    "default": ["I'm sorry, I don't understand that.", "Can you please rephrase?", "I'm still learning. Could you try a different question?"]
}

def get_response(user_input):
    """
    Analyzes the user's input and returns a suitable response.
    """
    user_input = user_input.lower()

    # Check for keywords and return a random response from the corresponding list
    if any(word in user_input for word in ["hello", "hi", "hey"]):
        return random.choice(responses["hello"])
    elif any(word in user_input for word in ["how are you", "how's it going"]):
        return random.choice(responses["how are you"])
    elif any(word in user_input for word in ["bye", "goodbye", "later"]):
        return random.choice(responses["bye"])
    else:
        return random.choice(responses["default"])

# Main chat loop
def chat():
    print("Bot: Hi! I'm a simple chatbot. Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("Bot: Goodbye!")
            break
        
        response = get_response(user_input)
        print(f"Bot: {response}")

# Start the chatbot
print("Welcome to the chatbot!")
if __name__ == "__main__":
    chat()