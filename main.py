# main.py

from data.data_loader import load_dual_dataset

ISOT_true_dataset = "data/ISOT Fake News Dataset/True.csv"
ISOT_fake_dataset = "data/ISOT Fake News Dataset/Fake.csv"

if __name__ == "__main__":
    # Troque os caminhos para testar outros datasets (ex: ISOT)
    df = load_dual_dataset(
        ISOT_true_dataset,
        ISOT_fake_dataset
    )

    print(df["label"].value_counts())
    print(df.head())
