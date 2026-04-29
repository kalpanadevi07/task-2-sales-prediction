def create_features(df):
    df['lag_1'] = df['sales'].shift(1)
    df['lag_7'] = df['sales'].shift(7)

    df['rolling_mean_7'] = df['sales'].rolling(window=7).mean()

    df = df.dropna()

    return df