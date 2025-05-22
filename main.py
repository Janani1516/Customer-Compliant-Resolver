import pandas as pd
import string
import nltk
import re
import spacy
import torch

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import pipeline, set_seed

# Download NLTK stopwords
nltk.download('stopwords')

# Load NLP tools
stop_words = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_md")  # Medium model with word vectors
stemmer = PorterStemmer()

# Text cleaning function
def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = text.split()
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    stemmed_words = [stemmer.stem(word) for word in words]  # Stemming
    doc = nlp(" ".join(stemmed_words))
    lemmatized_words = [token.lemma_ for token in doc]  # Lemmatization
    return " ".join(lemmatized_words)

# Load dataset
df = pd.read_json("synthetic_customer_complaints.json", lines=True)

# Clean complaints
df['Cleaned_Complaint'] = df['Complaint'].apply(clean_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['Cleaned_Complaint']).toarray()
y = df['Category']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n=== Model Accuracy on Test Set: {accuracy * 100:.2f}% ===")
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

# Category prediction
def predict_category(new_text):
    cleaned = clean_text(new_text)
    vectorized = vectorizer.transform([cleaned])
    prediction = rf_model.predict(vectorized)
    return prediction[0]

# Load FLAN-T5-Large model
print("\nLoading Transformer model (FLAN-T5 Large)... This may take time.")
device = 0 if torch.cuda.is_available() else -1
generator = pipeline("text2text-generation", model="google/flan-t5-large", device=device)
set_seed(42)

# Resolution generator
def generate_solution(complaint, category):
    prompt = (
        f"You are a customer service assistant.\n"
        f"Customer complaint: \"{complaint}\"\n"
        f"Issue category: {category}\n"
        f"Provide a polite and helpful resolution:"
    )
    try:
        result = generator(prompt, max_length=150, num_return_sequences=1, do_sample=True, temperature=0.7)
        return result[0]['generated_text']
    except Exception as e:
        return f"Error generating solution: {e}"

# Sample prediction
sample_df = df.head(10).copy()
sample_df['Predicted_Category'] = sample_df['Complaint'].apply(predict_category)
sample_df['Predicted_Solution'] = sample_df.apply(
    lambda row: generate_solution(row['Complaint'], row['Predicted_Category']), axis=1
)

print("\n=== Sample Dataset Output ===")
print(sample_df[['Complaint', 'Predicted_Category']])

# User input
print("\n=== User Input ===")
user_input = input("Enter a new customer complaint: ")

user_category = predict_category(user_input)
user_solution = generate_solution(user_input, user_category)

print("\nPredicted Category:", user_category)
print("Suggested Resolution:", user_solution)
