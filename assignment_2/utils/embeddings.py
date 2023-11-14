import pandas as pd


def save_embeddings_to_tsv(embeddings: any, path: str) -> None:
    df = pd.DataFrame(data=embeddings)
    df.to_csv(path, sep="\t", header=False, index=False)
