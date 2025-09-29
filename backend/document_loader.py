"""
document_loader.py

This module provides utility functions to preprocess an HR Q&A dataset
for ingestion into a vector database. It handles:
- Loading raw data from a CSV file.
- Cleaning and normalizing questions and answers.
"""

import pandas as pd
import os
import re
import pandas as pd
import spacy
import contractions
from nltk.corpus import stopwords

# Load spaCy model for lemmatization
nlp = spacy.load("en_core_web_sm")

STOPWORDS = set(stopwords.words('english'))

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the input CSV file into a pandas DataFrame.

    Args:
        file_path (str): Path to the input CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        FileNotFoundError: If the file does not exist at the provided path.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)


def clean_text(text: str, for_embeddings=True) -> str:
    """
    Clean and normalize a single text string for better embeddings and search.

    Args:
        text (str): Input text string.
        for_embeddings (bool): If False, will also remove stopwords (useful for BM25).
    
    Returns:
        str: Cleaned text string.
    """
    # Expand contractions: "can't" -> "cannot"
    text = contractions.fix(text)

    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # Lemmatization
    doc = nlp(text)
    text = " ".join([token.lemma_ for token in doc])

    # Optional: remove stopwords for BM25
    if not for_embeddings:
        text = " ".join([word for word in text.split() if word not in STOPWORDS])

    # Strip extra spaces
    return text.strip()


def clean_and_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced cleaning for Questions and Answers.
    """
    # Drop empty rows
    df = df.dropna(subset=["Question", "Answer"])

    # Deduplicate questions
    df = df.drop_duplicates(subset=["Question"])

    # Clean text
    df["Question"] = df["Question"].apply(lambda x: clean_text(x, for_embeddings=True))
    df["Answer"] = df["Answer"].apply(lambda x: clean_text(x, for_embeddings=True))

    return df

def preprocess_dataframe(file_path: str) -> pd.DataFrame:
    """
    Full preprocessing pipeline for HR Q&A data:
    1. Load the CSV dataset.
    2. Clean and normalize question and answer text.

    Args:
        file_path (str): Path to the raw input CSV file.

    Returns:
        pd.DataFrame: Fully processed and ready-to-use DataFrame.
    """
    df = load_data(file_path)
    df = clean_and_normalize(df)
    return df


def save_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save the processed DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): Processed HR Q&A dataset.
        output_path (str): Path to save the processed CSV file.

    Returns:
        None
    """
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to: {output_path}")
