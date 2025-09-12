import pandas as pd
import json

def load_data(path, format=''):
    if format == 'pandas':
        return pd.read_csv(path, encoding='utf-8', index_col=None)
    elif format == 'dict':
        return json.loads(path)