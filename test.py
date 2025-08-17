from modules.preprocessing import load_and_preprocess

# Correct dataset names and target columns
datasets = {
    "Diabetes": (r"D:\Health care project\data\diabetes.csv", "Outcome"),
    "Heart Disease": (r"D:\Health care project\data\Heart_disease_cleveland_new.csv", "target"),
}

for name, (path, target) in datasets.items():
    try:
        X, y, scaler = load_and_preprocess(path, target)
        print(f"✅ {name} dataset loaded successfully!")
        print(f"   Features shape: {X.shape}, Labels shape: {y.shape}\n")
    except Exception as e:
        print(f"❌ Error with {name} dataset: {e}\n")
