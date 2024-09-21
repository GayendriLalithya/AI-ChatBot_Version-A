from flask import Flask, render_template, request, jsonify
import pyttsx3
import speech_recognition as sr
import torch
import random
import json
from nltk.stem import WordNetLemmatizer
from utils.model_utils import load_model, load_label_encoder, load_vectorizer
from utils.data_utils import load_knowledge_base, find_product_details

app = Flask(__name__)

# Initialize pyttsx3 engine
engine = pyttsx3.init()

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()


# Tokenization and lemmatization function with print statements
def tokenize(text):
    tokens = text.split()  # Tokenize the text by splitting
    lemmatized_tokens = []
    for token in tokens:
        lemmatized_token = lemmatizer.lemmatize(token)
        print(f"Original token: {token}, Lemmatized token: {lemmatized_token}")
        lemmatized_tokens.append(lemmatized_token)
    return lemmatized_tokens


# Load model and preprocessors
model_path = 'models/intent_classifier.pth'
label_encoder_path = 'models/label_encoder.pkl'
vectorizer_path = 'models/vectorizer.pkl'
intents_path = 'data/intents.json'
knowledge_base_path = 'data/knowledge_base.json'

label_encoder = load_label_encoder(label_encoder_path)
vectorizer = load_vectorizer(vectorizer_path, tokenize)
with open(intents_path, 'r') as f:
    intents = json.load(f)['intents']
knowledge_base = load_knowledge_base(knowledge_base_path)

input_dim = len(vectorizer.get_feature_names_out())
output_dim = len(label_encoder.classes_)

model = load_model(model_path, input_dim, output_dim)
model.eval()


# Preprocess input function
def preprocess_input(user_input):
    X = vectorizer.transform([user_input]).toarray()
    inputs = torch.tensor(X, dtype=torch.float32)
    print(f"Preprocess input: {inputs}")
    return inputs


def predict_intent(user_input):
    preprocessed_input = preprocess_input(user_input)
    print(f"Preprocessed input: {preprocessed_input}")
    with torch.no_grad():
        output = model(preprocessed_input)
        print(f"Model Output: {output}")
    predicted_intent_index = torch.argmax(output, dim=1).item()
    print(f"Predicted Intent Index: {predicted_intent_index}")
    predicted_intent = label_encoder.classes_[predicted_intent_index]
    print(f"Predicted Intent: {predicted_intent}")
    return predicted_intent


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/speak', methods=['POST'])
def speak():
    text = request.json.get('text')
    engine.say(text)
    engine.runAndWait()  # Ensure this is run within proper error handling
    return jsonify({'status': 'success'})


@app.route('/listen', methods=['POST'])
def listen():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    try:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
            text = recognizer.recognize_google(audio)
            return jsonify({'text': text})
    except sr.UnknownValueError:
        return jsonify({'error': 'Could not understand the audio'})
    except sr.RequestError as e:
        return jsonify({'error': f'Could not request results; {e}'})


@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    tag = predict_intent(user_input)

    print(f"User Input: {user_input}")
    print(f"Predicted tag: {tag}")

    response = "Sorry, I couldn't understand your query."
    if tag == 'product_query':
        product_name = user_input.split("about")[-1].strip()  # Extract product name from user input
        response = find_product_details(product_name, knowledge_base)
    else:
        for intent in intents:
            if intent['tag'] == tag:
                response = random.choice(intent['responses'])
                break

    print(f"Response: {response}")
    return jsonify({'response': response})


def log_interaction(user_input, response):
    new_data = {
        "patterns": [user_input],
        "responses": [response],
        "tag": "user_generated"
    }
    try:
        with open('data/user_generated_intents.json', 'r') as file:
            user_generated_intents = json.load(file)
    except FileNotFoundError:
        user_generated_intents = {"intents": []}

    user_generated_intents['intents'].append(new_data)

    with open('data/user_generated_intents.json', 'w') as file:
        json.dump(user_generated_intents, file, indent=4)


if __name__ == '__main__':
    app.run(debug=True)
