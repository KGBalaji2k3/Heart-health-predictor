
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(file_path, target_column):

    df = pd.read_csv(file_path)

    df.fillna(df.mean(), inplace=True)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler


if __name__ == "__main__":

    try:
        X, y, scaler = load_and_preprocess(
            r"D:\Health care project\data\Heart_disease_cleveland_new.csv", "target"
        )
        print("✅ Heart dataset preprocessed!")
        print("   Features shape:", X.shape, "Labels shape:", y.shape, "\n")
    except Exception as e:
        print("❌ Error with Heart dataset:", e)

