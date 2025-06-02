from data_loader import load_dual_dataset
from preprocessor import apply_preprocessing
from vectorizer import vectorize_text
from trainer_nb import train_naive_bayes
from evaluator import evaluate_model
from sklearn.model_selection import train_test_split
from trainer_svm import train_svm
from comparator import evaluate_and_compare
from comparator import cross_validate_models


# Troque os caminhos para testar outros datasets (ex: ISOT/...)
ISOT_true_dataset = "ISOT Fake News Dataset/True.csv"
ISOT_fake_dataset = "ISOT Fake News Dataset/Fake.csv"

if __name__ == "__main__":
    #load dataset
    df = load_dual_dataset(ISOT_true_dataset, ISOT_fake_dataset)

    #pre processing
    df = apply_preprocessing(df, text_column="text")

    #vetorization
    X, vectorizer = vectorize_text(df["text_clean"])
    y = df["label"]

    num_fake = df[df["label"] == 0].shape[0]
    num_real = df[df["label"] == 1].shape[0]

    print(f"\nTotal de notícias falsas: {num_fake}")
    print(f"Total de notícias reais: {num_real}")
    print(f"Total geral: {df.shape[0]}")

    print("\n📦 Primeiras linhas do dataset original:")
    print(df[["label", "text"]].head())

    print("\n🧹 Primeiras linhas do texto limpo (pré-processado):")
    print(df[["label", "text_clean"]].head())

    #show vetorization example
    print("\n🔍 Exemplo de vetorização TF-IDF das 5 primeiras notícias:\n")

    feature_names = vectorizer.get_feature_names_out()

    for i in range(5):
        titulo = df["title"].iloc[i]
        label_valor = y.iloc[i]
        print(f"\n📰 Título: {titulo}\n🔖 Classe: {'Fake' if label_valor == 0 else 'Real'}")

        row = X[i].toarray().flatten()
        nonzero_indices = row.nonzero()[0]

        for idx in nonzero_indices[:10]:  # mostra os 10 primeiros termos
            palavra = feature_names[idx]
            peso = row[idx]
            print(f"  {palavra}: {peso:.4f}")
    
    #application of NB
    print("\n🚀 Treinando modelo Naive Bayes...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = train_naive_bayes(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    #application of SVM
    print("\n🚀 Treinando modelo SVM...")
    svm_model = train_svm(X_train, y_train)
    evaluate_model(svm_model, X_test, y_test)

    #compare models
    nb_model = train_naive_bayes(X_train, y_train)
    svm_model = train_svm(X_train, y_train)

    evaluate_and_compare(
        models=[nb_model, svm_model],
        model_names=["Naive Bayes", "SVM"],
        X_test=X_test,
        y_test=y_test
    )

    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import LinearSVC
    #cross validation
    cross_validate_models(
        models=[MultinomialNB(), LinearSVC()],
        model_names=["Naive Bayes", "SVM"],
        X=X,
        y=y,
        cv=5
    )
