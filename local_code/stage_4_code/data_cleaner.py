# Required imports for file handling, text processing, and data serialization
import os
import pickle
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import json
# punkt: for tokenization, essentially spliting a text into individual words
# stopwords: for removing common words

# Download only the required NLTK resources

nltk.download('punkt')

nltk.download('stopwords')

nltk.download('punkt_tab')

nltk.download('tokenizers/punkt/english.pickle')
def clean_text(text):
    """Clean text by removing punctuation, converting to lowercase, and removing stopwords"""
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert to lowercase and tokenize
    words = word_tokenize(text.lower())
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    cleaned_text = [word for word in words if word not in stop_words]
    return ' '.join(cleaned_text)

def clean_generation_text(text):
    """Clean text for generation while preserving structure and meaning"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def clean_and_save_classification_data():
    """Clean classification data and save to pickle file"""
    result = {
        "train": {"pos": [], "neg": []},
        "test": {"pos": [], "neg": []}
    }
    
    base_path = './data/stage_4_data/text_classification'
    
    # Process both training and test data
    for split in ['train', 'test']:
        for sentiment in ['pos', 'neg']:
            path = os.path.join(base_path, split, sentiment)
            for filename in os.listdir(path):
                print(f"Processing {filename} {split} {sentiment} data...")
                with open(os.path.join(path, filename), 'r', encoding='utf-8') as f:
                    text = f.read()
                    cleaned_text = clean_text(text)
                    result[split][sentiment].append(cleaned_text)
    
    # Save cleaned data
    with open('./data/stage_4_data/cleaned_classification.pkl', 'wb') as f:
        pickle.dump(result, f)
    
    with open('./data/stage_4_data/cleaned_classification.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    print("Classification data cleaned and saved:")
    print(f"Train positive: {len(result['train']['pos'])} reviews")
    print(f"Train negative: {len(result['train']['neg'])} reviews")
    print(f"Test positive: {len(result['test']['pos'])} reviews")
    print(f"Test negative: {len(result['test']['neg'])} reviews")

def clean_and_save_generation_data():
    """Clean generation data and save to pickle file"""
    with open('./data/stage_4_data/text_generation/data', 'r', encoding='utf-8') as f:
        text = f.read()
    
    cleaned_text = clean_generation_text(text)
    
    # Save cleaned data
    with open('./data/stage_4_data/cleaned_generation.pkl', 'wb') as f:
        pickle.dump(cleaned_text, f)
    
    print("\nGeneration data cleaned and saved:")
    print(f"Total characters: {len(cleaned_text)}")
    print(f"Unique characters: {len(set(cleaned_text))}")

clean_and_save_classification_data()
clean_and_save_generation_data()
