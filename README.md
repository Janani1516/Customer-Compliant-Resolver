# Customer-Complain-Resolver
An AI-powered system that classifies customer complaints into categories using Random Forest and generates resolutions with FLAN-T5. It cleans and processes text with NLP techniques and supports both batch and real-time complaint handling.

# AI Customer Complaint Resolver

This project classifies customer complaints into categories and generates suggested resolutions using a combination of classical machine learning (Random Forest) and transformer-based models (FLAN-T5).

## Features
- Cleans and preprocesses complaint text (lowercasing, removing numbers/punctuation, stemming, lemmatization).
- Classifies complaints into predefined categories using a Random Forest classifier trained on TF-IDF features.
- Uses Google's FLAN-T5 transformer to generate a natural language resolution for the complaint.
- Allows dynamic user input for real-time predictions.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/ai-customer-complaint-resolver.git
cd ai-customer-complaint-resolver

2.Install the required Python packages:

pip install -r requirements.txt

3.Download the necessary spaCy model:

python -m spacy download en_core_web_md

4.Run the file:

python main.py


**If using Jupyter Notebook or Google Colab ensure u installed the required packages and upload the dataset to the run time. Now run the code.**
