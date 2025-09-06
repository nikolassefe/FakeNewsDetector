import tkinter as tk
from tkinter import messagebox
import pickle
import re

#Load Model and Vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

#Text Cleaning Function
def clean_text(text):
    text = re.sub('[^a-zA-Z ]', '', text)
    return text.lower()

#Predict Function
def predict():
    title = entry.get()
    cleaned = clean_text(title)
    vector = vectorizer.transform([cleaned]).toarray()
    pred = model.predict(vector)[0]
    result = "Fake news ❌" if pred == 1 else "Real news ✅"
    messagebox.showinfo("Prediction", result)

#GUI SetUp
root = tk.Tk()
root.title("Fake News Detector")

tk.Label(root, text="Enter news title:").pack(pady=5)
entry = tk.Entry(root, width=50)
entry.pack(pady=5)

tk.Button(root, text="Check", command=predict).pack(pady=10)

root.mainloop()
