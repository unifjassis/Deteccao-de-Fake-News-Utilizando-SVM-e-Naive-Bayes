import pandas as pd

def load_dual_dataset(true_path: str, fake_path: str) -> pd.DataFrame:
    df_true = pd.read_csv(true_path)
    df_fake = pd.read_csv(fake_path)

    df_true["label"] = 1
    df_fake["label"] = 0

    df = pd.concat([df_true, df_fake], ignore_index=True)
    #print(f"Total de amostras: {df.shape[0]}")
    return df
