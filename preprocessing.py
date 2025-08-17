# modules/preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(file_path, target_column):
    """
    Loads dataset, handles missing values, scales features, 
    and splits into X (features) and y (labels).
    """
    # Load dataset
    df = pd.read_csv(file_path)

    # Handle missing values (replace with column mean)
    df.fillna(df.mean(), inplace=True)

    # Features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler


if __name__ == "__main__":
    # Test on Diabetes dataset
    try:
        X, y, scaler = load_and_preprocess(
            r"D:\Health care project\data\diabetes.csv", "Outcome"
        )
        print("✅ Diabetes dataset preprocessed!")
        print("   Features shape:", X.shape, "Labels shape:", y.shape, "\n")
    except Exception as e:
        print("❌ Error with Diabetes dataset:", e)

    # Test on Heart dataset
    try:
        X, y, scaler = load_and_preprocess(
            r"D:\Health care project\data\Heart_disease_cleveland_new.csv", "target"
        )
        print("✅ Heart dataset preprocessed!")
        print("   Features shape:", X.shape, "Labels shape:", y.shape, "\n")
    except Exception as e:
        print("❌ Error with Heart dataset:", e)
