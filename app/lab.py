# Importing NumPy
import numpy as np

# Task 1 & 2
empty_array = np.empty((3, 3))
full_array = np.full((3, 3), fill_value=7)  # example full value
random_array = np.random.randn(25)

# Task 3
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
dot_product = np.dot(a, b)

# Task 4
arr = np.array([[3, 7, 1], [10, 3, 2], [5, 6, 7]])
sorted_row = np.sort(arr, axis=1)
sorted_col = np.sort(arr, axis=0)
sorted_all = np.sort(arr, axis=None)

# Task 5
array_list = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
means = [np.mean(arr) for arr in array_list]

# Task 6
languages = np.array(['PHP C# Python C Java C++'])
split_languages = np.char.split(languages)


# Importing Matplotlib and Pandas
import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV data
data = pd.read_csv('company_sales_data.csv')

# Task 7
plt.figure(figsize=(10, 5))
plt.plot(data['total_profit'])
plt.xlabel('Month Number')
plt.ylabel('Total profit')
plt.title('Total Profit per Month')
plt.grid(True)
plt.show()

# Task 8
plt.figure(figsize=(10, 5))
plt.plot(data['total_profit'], linestyle=':', color='red', label='Total Profit', marker='o', linewidth=3)
plt.xlabel('Month Number')
plt.ylabel('Sold units number')
plt.title('Styled Total Profit Plot')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# Task 8.i - Multiline Plot
plt.figure(figsize=(12, 6))
products = data.columns[1:-1]  # Skipping 'month_number' and 'total_profit'
for product in products:
    plt.plot(data['month_number'], data[product], label=product)
plt.xlabel('Month Number')
plt.ylabel('Units Sold')
plt.title('Units Sold per Month for Each Product')
plt.legend()
plt.grid(True)
plt.show()

# Task 9 - Toothpaste sales with grid
plt.figure(figsize=(10, 5))
plt.plot(data['month_number'], data['toothpaste'], marker='o')
plt.xlabel('Month Number')
plt.ylabel('Toothpaste Units Sold')
plt.title('Toothpaste Sales Data')
plt.grid(True, linestyle='--')
plt.show()

# Task 10 - Bar chart for Face Cream and Face Wash
bar_width = 0.4
month = data['month_number']
plt.figure(figsize=(10, 5))
plt.bar(month - 0.2, data['facecream'], width=bar_width, label='Face Cream', align='center')
plt.bar(month + 0.2, data['facewash'], width=bar_width, label='Face Wash', align='center')
plt.xlabel('Month Number')
plt.ylabel('Units Sold')
plt.title('Sales of Face Cream and Face Wash')
plt.legend()
plt.show()



## assignement 2


# Import Required Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris Dataset
df = pd.read_csv('iris.csv')

# a. Show size of the dataset
print("Dataset size (rows, columns):", df.shape)

# b. Show datatype for each column
print("\nData types for each column:\n", df.dtypes)

# c. Show distribution of data
print("\nSummary statistics:\n", df.describe())

# d. Check for null values
print("\nNull values in each column:\n", df.isnull().sum())

# e. Check for duplicates
print("\nNumber of duplicate rows:", df.duplicated().sum())

# f. Instances per species
print("\nNumber of instances per species:\n", df['species'].value_counts())

# g. Compare sepal length and sepal width
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x='sepal_length', y='sepal_width', hue='species')
plt.title("Sepal Length vs Sepal Width")
plt.show()

# h. Compare petal length and petal width
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x='petal_length', y='petal_width', hue='species')
plt.title("Petal Length vs Petal Width")
plt.show()

# i. Pairplot to show all comparisons
sns.pairplot(df, hue='species')
plt.suptitle("Pairplot of Iris Dataset", y=1.02)
plt.show()

# j. Histograms of all features
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
df[features].hist(figsize=(10, 8), color='skyblue', edgecolor='black')
plt.suptitle("Histograms of Features", y=1.02)
plt.show()

# k. Boxplot for each feature across species
plt.figure(figsize=(12, 8))
for i, feature in enumerate(features):
    plt.subplot(2, 2, i+1)
    sns.boxplot(data=df, x='species', y=feature)
    plt.title(f'Boxplot of {feature}')
plt.tight_layout()
plt.show()

# l. Violinplot for each feature across species
plt.figure(figsize=(12, 8))
for i, feature in enumerate(features):
    plt.subplot(2, 2, i+1)
    sns.violinplot(data=df, x='species', y=feature)
    plt.title(f'Violinplot of {feature}')
plt.tight_layout()
plt.show()



















##pip install requests beautifulsoup4

# news_sentiment_labeler.py

import requests
from bs4 import BeautifulSoup
import re

# -------- VERSION --------
__version__ = "1.0.0"

# -------- SCRAPER MODULE --------
def fetch_moneycontrol_news(url: str) -> list:
    """Scrapes news headlines/articles from Moneycontrol."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        headlines = []
        for item in soup.select('h2, h3, p'):
            text = item.get_text(strip=True)
            if text:
                headlines.append(text)
        
        return headlines
    except requests.exceptions.RequestException as e:
        print("Error fetching news:", e)
        return []

# -------- CLEANER MODULE --------
def clean_text(text: str) -> str:
    """Cleans the input text for sentiment processing."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9.,!?\' ]', '', text)
    return text.strip()

# -------- MAIN FUNCTION --------
def main():
    print(f"📰 News Sentiment Labeling Script — Version {__version__}")
    url = "https://www.moneycontrol.com/news/"
    
    print("\nFetching articles from Moneycontrol...")
    raw_articles = fetch_moneycontrol_news(url)
    print(f"✅ Fetched {len(raw_articles)} articles.")

    print("\nCleaning articles...")
    cleaned_articles = [clean_text(article) for article in raw_articles]

    print("\n🔎 Sample Cleaned Articles:")
    for i, article in enumerate(cleaned_articles[:10], 1):
        print(f"{i}. {article}")

if __name__ == "__main__":
    main()

















pip install pandas numpy scikit-learn matplotlib seaborn wordcloud tensorflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, Bidirectional

# -------- Data Loader --------
def load_dataset(filepath):
    return pd.read_csv(filepath)

# -------- TF-IDF Model --------
def tfidf_model(X_train, X_test, y_train, y_test):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)

    print("\nTF-IDF Model Performance:")
    print(classification_report(y_test, y_pred))
    return y_pred

# -------- LSTM/GRU/BiLSTM Model --------
def keras_rnn_model(X_train, X_test, y_train, y_test, rnn_type='lstm', bidirectional=False):
    tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    max_len = max(len(seq) for seq in X_train_seq)
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')

    model = Sequential()
    model.add(Embedding(input_dim=5000, output_dim=64, input_length=max_len))

    if rnn_type.lower() == 'gru':
        rnn_layer = GRU(64)
    else:
        rnn_layer = LSTM(64)

    if bidirectional:
        model.add(Bidirectional(rnn_layer))
    else:
        model.add(rnn_layer)

    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train_pad, y_train, epochs=3, batch_size=32, validation_split=0.1)
    y_pred_prob = model.predict(X_test_pad)
    y_pred = (y_pred_prob > 0.5).astype("int32")

    print(f"\n{rnn_type.upper()} Model Performance:")
    print(classification_report(y_test, y_pred))
    return y_pred

# -------- Evaluation Visualization --------
def plot_confusion(y_test, y_pred, title='Confusion Matrix'):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# -------- WordCloud Generator --------
def show_wordcloud(df, column='text'):
    text = ' '.join(df[column].astype(str))
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title("Word Cloud of Tweets")
    plt.show()

# -------- Main Function --------
def main():
    df = load_dataset('data/twitter_sentiment_dataset.csv')
    print(f"Dataset Loaded: {df.shape} rows")

    X = df['text']
    y = df['label']  # Binary sentiment: 0/1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # TF-IDF + Logistic Regression
    tfidf_pred = tfidf_model(X_train, X_test, y_train, y_test)
    plot_confusion(y_test, tfidf_pred, title="TF-IDF Confusion Matrix")

    # LSTM
    lstm_pred = keras_rnn_model(X_train, X_test, y_train, y_test, rnn_type='lstm')
    plot_confusion(y_test, lstm_pred, title="LSTM Confusion Matrix")

    # Bi-LSTM
    bilstm_pred = keras_rnn_model(X_train, X_test, y_train, y_test, rnn_type='lstm', bidirectional=True)
    plot_confusion(y_test, bilstm_pred, title="BiLSTM Confusion Matrix")

    # GRU
    gru_pred = keras_rnn_model(X_train, X_test, y_train, y_test, rnn_type='gru')
    plot_confusion(y_test, gru_pred, title="GRU Confusion Matrix")

    # WordCloud
    show_wordcloud(df)

if __name__ == "__main__":
    main()














# opencv-python
# Pillow
# numpy

import cv2
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
import threading

# Load YOLOv3 or use pretrained MobileNet SSD for object detection
net = cv2.dnn.readNetFromCaffe(
    "models/MobileNetSSD_deploy.prototxt",
    "models/MobileNetSSD_deploy.caffemodel"
)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Object + Face Detection Logic
def detect_objects_and_faces(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Detect Faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,wf,hf) in faces:
        cv2.rectangle(frame, (x, y), (x + wf, y + hf), (255, 0, 0), 2)
        cv2.putText(frame, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Detect Other Living Objects
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]
            if label in ["person", "cat", "dog", "horse", "sheep", "cow", "bird"]:
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, label, (startX, startY-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

# --------- Simple Tkinter UI ---------
class App:
    def __init__(self, window):
        self.window = window
        self.window.title("Face & Living Object Detector")
        self.label = Label(window)
        self.label.pack()

        self.btn = Button(window, text="Start Detection", command=self.start_video)
        self.btn.pack()

        self.video_capture = cv2.VideoCapture(0)
        self.running = False

    def start_video(self):
        self.running = True
        thread = threading.Thread(target=self.update)
        thread.start()

    def update(self):
        while self.running:
            ret, frame = self.video_capture.read()
            if ret:
                frame = detect_objects_and_faces(frame)
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = ImageTk.PhotoImage(Image.fromarray(img))
                self.label.config(image=img)
                self.label.image = img

    def on_closing(self):
        self.running = False
        self.video_capture.release()
        self.window.destroy()

# Run App
if __name__ == "__main__":
    window = tk.Tk()
    app = App(window)
    window.protocol("WM_DELETE_WINDOW", app.on_closing)
    window.mainloop()






