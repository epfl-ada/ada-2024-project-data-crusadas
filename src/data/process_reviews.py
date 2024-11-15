import re
import gzip
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from multiprocessing import Pool
from tqdm import tqdm

# Initialize lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to filter by part of speech
def filter_by_pos(tokens, allowed_tags={'NN', 'JJ'}):  # Retain nouns (NN) and adjectives (JJ)
    pos_tags = pos_tag(tokens)
    return [word for word, tag in pos_tags if tag in allowed_tags]

# Preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenize text
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # Lemmatize and remove stopwords
    return filter_by_pos(tokens)

# Batch processing function (top-level definition)
def process_batch(batch_lines):
    batch_results = []
    for line in batch_lines:
        if 'text:' in line:
            review_text = line.split('text:')[-1].strip()
            review_text = re.sub(r'\d+', '', review_text)  # Remove numbers
            tokens = preprocess_text(review_text)
            batch_results.append(tokens)
    return batch_results


def get_tokens(file_path, review_limit=10000):
    all_tokens = []
    batch_size = 1000
    batch = []
    count = 0

    with gzip.open(file_path, 'rt', encoding='utf-8') as file:
        for line in file:
            if count >= review_limit:
                break
            batch.append(line)
            count += 1

            if len(batch) >= batch_size:
                # Parallelize batch processing
                with Pool() as pool:
                    results = pool.map(process_batch, [batch])
                for result in results:
                    all_tokens.extend(result)
                batch = []

        # Process remaining lines in the last batch
        if batch:
            with Pool() as pool:
                results = pool.map(process_batch, [batch])
            for result in results:
                all_tokens.extend(result)

    return all_tokens

