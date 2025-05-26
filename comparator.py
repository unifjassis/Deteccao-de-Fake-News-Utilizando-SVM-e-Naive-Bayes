from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, cross_validate
import numpy as np

def evaluate_and_compare(models, model_names, X_test, y_test, target_names=["Fake", "Real"]):
    results = {}

    for model, name in zip(models, model_names):
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)

        results[name] = {
            "accuracy": acc,
            "precision": report["Fake"]["precision"],
            "recall": report["Fake"]["recall"],
            "f1-score": report["Fake"]["f1-score"]
        }

        # Terminal output
        print(f"\n📊 Avaliação para: {name}")
        print(f"Acurácia: {acc:.4f}")
        print(f"Precision (Fake): {report['Fake']['precision']:.4f}")
        print(f"Recall (Fake):    {report['Fake']['recall']:.4f}")
        print(f"F1-Score (Fake):  {report['Fake']['f1-score']:.4f}")

    """
        # Gráfico comparativo
        labels = list(results.keys())
        accs = [results[m]["accuracy"] for m in labels]
        precisions = [results[m]["precision"] for m in labels]
        recalls = [results[m]["recall"] for m in labels]
        f1s = [results[m]["f1-score"] for m in labels]

        x = range(len(labels))
        width = 0.2

        plt.figure(figsize=(10, 6))
        plt.bar([i - width for i in x], accs, width=width, label="Acurácia")
        plt.bar(x, precisions, width=width, label="Precisão")
        plt.bar([i + width for i in x], recalls, width=width, label="Revocação")
        plt.bar([i + 2*width for i in x], f1s, width=width, label="F1-score")

        plt.xticks(x, labels)
        plt.ylim(0, 1.05)
        plt.title("Comparação de Desempenho dos Modelos")
        plt.legend()
        plt.grid(True, axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()
    """

def cross_validate_models(models, model_names, X, y, cv=5):
    """
    Aplica validação cruzada (k-fold) em múltiplos modelos e exibe métricas médias.
    """
    print(f"\n📊 Validação cruzada ({cv}-fold):\n")
    results = {}

    for model, name in zip(models, model_names):
        print(f"🔹 Modelo: {name}")
        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=["accuracy", "precision", "recall", "f1"],
            return_train_score=False
        )

        results[name] = {
            "accuracy": np.mean(scores["test_accuracy"]),
            "precision": np.mean(scores["test_precision"]),
            "recall": np.mean(scores["test_recall"]),
            "f1": np.mean(scores["test_f1"])
        }

        print(f"  Acurácia média:  {results[name]['accuracy']:.4f}")
        print(f"  Precisão média:  {results[name]['precision']:.4f}")
        print(f"  Revocação média: {results[name]['recall']:.4f}")
        print(f"  F1-score média:  {results[name]['f1']:.4f}\n")

    # Gráfico comparativo
    labels = list(results.keys())
    accs = [results[m]["accuracy"] for m in labels]
    precisions = [results[m]["precision"] for m in labels]
    recalls = [results[m]["recall"] for m in labels]
    f1s = [results[m]["f1"] for m in labels]

    x = range(len(labels))
    width = 0.2

    plt.figure(figsize=(10, 6))
    plt.bar([i - width for i in x], accs, width=width, label="Acurácia")
    plt.bar(x, precisions, width=width, label="Precisão")
    plt.bar([i + width for i in x], recalls, width=width, label="Revocação")
    plt.bar([i + 2*width for i in x], f1s, width=width, label="F1-score")

    plt.xticks(x, labels)
    plt.ylim(0, 1.05)
    plt.title("Comparação de Desempenho com Validação Cruzada")
    plt.legend()
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("val_cross_modelos.png")
    print("📈 Gráfico salvo como 'val_cross_modelos.png'")

