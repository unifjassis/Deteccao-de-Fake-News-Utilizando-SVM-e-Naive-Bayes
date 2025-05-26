from sklearn.svm import LinearSVC

def train_svm(X_train, y_train, C=1.0):
    """
    Treina um modelo SVM linear para classificação binária.

    Args:
        X_train: matriz TF-IDF de treinamento
        y_train: rótulos de treinamento
        C (float): parâmetro de regularização

    Returns:
        modelo treinado
    """
    model = LinearSVC(C=C)
    model.fit(X_train, y_train)
    return model
