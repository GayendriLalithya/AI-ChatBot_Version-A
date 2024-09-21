import nltk
import json
from sklearn.preprocessing import LabelEncoder

nltk.download('punkt')


def preprocess_data(intents):
    patterns = []
    tags = []
    responses = {}

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            tokenized_pattern = nltk.word_tokenize(pattern)
            patterns.append(tokenized_pattern)
            tags.append(intent['tag'])
        responses[intent['tag']] = intent['responses']

    return patterns, tags, responses


def load_intents():
    with open('data/intents.json', 'r') as file:
        original_intents = json.load(file)

    try:
        with open('data/user_generated_intents.json', 'r') as file:
            user_intents = json.load(file)
    except FileNotFoundError:
        user_intents = {"intents": []}

    combined_intents = {
        "intents": original_intents['intents'] + user_intents['intents']
    }

    return combined_intents
