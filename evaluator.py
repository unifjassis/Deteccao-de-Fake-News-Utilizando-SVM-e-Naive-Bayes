from sklearn.metrics import classification_report, accuracy_score

def evaluate_model(model, X_test, y_test, target_names=["Fake", "Real"]):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("\n📈 Resultados do modelo:")
    print(f"Acurácia: {acc:.4f}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred, target_names=target_names))
