import pandas as pd
from sklearn.model_selection import train_test_split
from src.trainer import get_trainer

# Load small sample
df = pd.read_csv("data/creditcard.csv", nrows=10000)

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Test RandomForest
trainer = get_trainer("random_forest", n_estimators=10, max_depth=3)
model = trainer.train(X_train, y_train)
metrics = trainer.evaluate(model, X_test, y_test)
print("\nRandomForest metrics:")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")

# Test wrong model type
try:
    get_trainer("xgboost")
except ValueError as e:
    print(f"\n{e}")