import pickle


def dump_pickle(path: str, obj: object) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str) -> any:
    with open(path, "rb") as f:
        return pickle.load(f)
