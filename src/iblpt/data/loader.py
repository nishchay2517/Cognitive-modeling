import pandas as pd


def load_dataset(path: str):
    cols = ['id','val_high','p_high','val_low','val_safe','sure','d1','mode']
    return pd.read_csv(path, sep=r'\s+', header=None, names=cols)



