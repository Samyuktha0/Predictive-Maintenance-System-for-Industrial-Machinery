import pandas as pd

def preprocess_data(filepath):
    # Load dataset
    df = pd.read_csv(filepath, sep=" ", header=None)

    # Add column names for clarity
    df.columns = [...] # Define your column names

    # Perform feature engineering, e.g., create RUL (Remaining Useful Life)
    # This is a key step for predictive maintenance
    df['RUL'] = df.groupby('unit_id')['time_in_cycles'].transform('max') - df['time_in_cycles']

    # Normalize data for model training
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    return df_normalized

if __name__ == '__main__':
    data = preprocess_data('data/NASA_turbofan_data.csv')
    data.to_csv('data/processed_data.csv', index=False)
