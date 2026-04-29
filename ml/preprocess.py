import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess(df):
    df['date'] = pd.to_datetime(df['date'])

    df = df.sort_values(by='date')

    df.fillna(method='ffill', inplace=True)

    return df