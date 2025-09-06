import os
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import numpy as np

#Setup Kaggle API
api = KaggleApi()
api.authenticate()
print("Kaggle authentication successful")

#Folder where files will be stored
data_dir = 'data'
os.makedirs(data_dir, exist_ok=True)

#Only download if files are missing
if not os.path.exists(os.path.join(data_dir, 'Fake.csv')) or not os.path.exists(os.path.join(data_dir, 'True.csv')):
    print("Downloading dataset from Kaggle...")
    api.dataset_download_files('clmentbisaillon/fake-and-real-news-dataset', path=data_dir, unzip=True)
else:
    print("Dataset already exists, skipping download.")

#Load CSVs
fake = pd.read_csv(os.path.join(data_dir, 'Fake.csv'))
true = pd.read_csv(os.path.join(data_dir, 'True.csv'))

fake['label'] = 1
true['label'] = 0
data = pd.concat([fake, true], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
data['text'] = data['title']

#NLTK SetUp
nltk.data.path.append('/Users/nikolasseferiadis/Documents/Python/Projects/Fake_News_Detection/nltk_data')
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub('[^a-zA-Z ]', '', text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text if word not in stop_words]
    return ' '.join(text)

#Clean text
cleaned_texts = []
for text in tqdm(data['text'], desc="Cleaning text", ascii=True):
    cleaned_texts.append(clean_text(text))
data['cleaned_text'] = cleaned_texts

#Vectorization in chunks with progress bar
vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
chunk_size = 100
vectorized_chunks = []

for i in tqdm(range(0, len(data['cleaned_text']), chunk_size), desc="Vectorizing text", ascii=True):
    chunk = data['cleaned_text'].iloc[i:i+chunk_size]
    if i == 0:
        vectorized_chunks.append(vectorizer.fit_transform(chunk).toarray())
    else:
        vectorized_chunks.append(vectorizer.transform(chunk).toarray())

X = np.vstack(vectorized_chunks)
y = data['label']

#Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train RandomForest tree by tree with progress bar
model = RandomForestClassifier(n_estimators=200, random_state=42, warm_start=True)

for n in tqdm(range(1, 201), desc="Training RandomForest", ascii=True):
    model.n_estimators = n
    model.fit(X_train, y_train)

#Predict test set with progress
y_pred = []
for x in tqdm(X_test, desc="Predicting test set", ascii=True):
    y_pred.append(model.predict([x])[0])

print(f"\nTraining finished")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

#Save model and vectorizer
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("\nModel and vectorizer saved.")
print("\nFake News Detector Ready.")
