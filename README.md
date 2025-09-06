# Fake News Detector

**Fake News Detector** is a Python project that classifies news headlines as real or fake using machine learning. 
The project leverages natural language processing (NLP) techniques, including text cleaning, stemming, 
and TF-IDF vectorization, combined with a Random Forest classifier.

## Features

- Downloads datasets automatically from Kaggle.  
- Cleans and preprocesses text data (removes punctuation, stopwords, and applies stemming).  
- Uses TF-IDF vectorization for feature extraction.  
- Trains a Random Forest classifier to predict fake vs real news.  
- Provides both **command-line interface (CLI)** and **GUI interface** using Tkinter.  
- Saves trained model and vectorizer for reuse.  

## Installation

1. Clone the repository:

```bash
git clone <your-repo-url>
cd FakeNewsDetector
```

2. Install required Python modules:

```bash
pip install -r requirements.txt
```

## Usage

### Command-Line Interface

```bash
python src/main.py
```

- Downloads datasets (if missing)  
- Trains the model  
- Allows interactive fake news detection  

### Graphical User Interface

```bash
python src/gui.py
```

- Enter news titles in the GUI to check if they are fake or real  

## Requirements

- Python 3.10+  
- `pandas`  
- `scikit-learn`  
- `nltk`  
- `tqdm`  
- `tkinter`  

> All dependencies can be installed via `pip install -r requirements.txt`.  

## Notes

- The model and vectorizer are saved locally after training (`model.pkl`, `vectorizer.pkl`).  
- Large files like models and NLTK datasets are **not included in the repository** to keep it lightweight. 
, They are downloaded or generated automatically.