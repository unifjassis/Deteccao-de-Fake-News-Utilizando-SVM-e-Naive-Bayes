# src/data_loader.py

import pandas as pd
from pathlib import Path

def load_dual_dataset(true_path: str, fake_path: str) -> pd.DataFrame:
    """
    Lê dois arquivos CSV separados (notícias verdadeiras e falsas),
    adiciona a coluna 'label' e concatena os dois em um único DataFrame.

    Args:
        true_path (str): Caminho para o arquivo CSV com notícias verdadeiras.
        fake_path (str): Caminho para o arquivo CSV com notícias falsas.

    Returns:
        pd.DataFrame: DataFrame combinado com coluna 'label' (1=real, 0=fake)
    """
    df_true = pd.read_csv(true_path)
    df_fake = pd.read_csv(fake_path)

    df_true["label"] = 1
    df_fake["label"] = 0

    df = pd.concat([df_true, df_fake], ignore_index=True)
    print(f"Total de amostras: {df.shape[0]}")
    return df
