from data_loader import load_dual_dataset
from preprocessor import apply_preprocessing
from vectorizer import vectorize_text

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

    print(f"\n🔢 Forma da matriz TF-IDF: {X.shape}")
    print("👉 Cada linha é um texto. Cada coluna é uma palavra.")

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