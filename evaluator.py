from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_test, y_test, model_name="Modelo"):
    """
    Avalia o modelo treinado, exibe métricas e gera matriz de confusão.
    """
    print(f"\n📈 Resultados do {model_name}:")

    # Faz previsões
    y_pred = model.predict(X_test)

    # Exibe métricas
    acc = accuracy_score(y_test, y_pred)
    print(f"Acurácia: {acc:.4f}\n")
    print("Relatório de Classificação:")
    print(classification_report(y_test, y_pred))

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Imprime matriz com legendas no terminal
    print("\n🔹 Matriz de Confusão (valores reais):")
    print(f"TP (True Positive)  - Previsto Fake / Real Fake: {tp}")
    print(f"TN (True Negative)  - Previsto Real / Real Real: {tn}")
    print(f"FP (False Positive) - Previsto Fake / Real Real: {fp}")
    print(f"FN (False Negative) - Previsto Real / Real Fake: {fn}")

    # Gera imagem da matriz
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", cbar=False,
                xticklabels=['Fake', 'Real'],
                yticklabels=['Fake', 'Real'])
    plt.xlabel('Previsto')
    plt.ylabel('Real')
    plt.title(f'Matriz de Confusão - {model_name}')
    plt.tight_layout()
    plt.savefig(f"matriz_confusao_{model_name}.png")
    plt.close()
