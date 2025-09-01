import pandas as pd

def read_data(filepath):
    return pd.read_csv(filepath, encoding='utf-8')
