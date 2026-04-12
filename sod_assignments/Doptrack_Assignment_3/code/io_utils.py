import pickle
import os

def save_results(results, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(results, f)


def load_results(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)