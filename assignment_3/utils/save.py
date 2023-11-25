import pandas as pd


def to_tsv(data: any, path: str) -> None:
    df = pd.DataFrame(data=data)
    df.to_csv(path, sep="\t", header=False, index=False)
