from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_text(corpus, max_features=5000):
    """
    Aplica vetorização TF-IDF ao corpus de textos limpos.

    Args:
        corpus (list): lista de textos (ex: df["text_clean"])
        max_features (int): máximo de palavras vetorizadas

    Returns:
        X (sparse matrix): matriz TF-IDF
        vectorizer: o objeto do TF-IDF treinado
    """
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer
