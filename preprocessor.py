import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
nltk.download("punkt")

stop_words = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    """
    Pré-processaa o texto com:
    - minúsculas
    - remoção de símbolos e números
    - remoção de stopwords
    - tokenização simplificada com split()
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)  # remove pontuação
    text = re.sub(r"\d+", " ", text)      # remove números
    tokens = text.split()  # tokeniza com base em espaços

    filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(filtered_tokens)

def apply_preprocessing(df, text_column="text"):
    df["text_clean"] = df[text_column].astype(str).apply(clean_text)
    return df
