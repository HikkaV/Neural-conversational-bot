import json

def load_json(path):
    with open(path, 'r') as f:
        file = json.load(f)
    return file