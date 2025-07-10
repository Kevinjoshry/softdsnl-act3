# train_model.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

df = pd.read_csv("my_dataset.csv")

sns.scatterplot(data=df, x="engine_size", y="horsepower", hue="car_type")
plt.title("Custom Dataset")
plt.show()

X = df[["engine_size", "horsepower"]]
le = LabelEncoder()
y = le.fit_transform(df["car_type"])

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "model.pkl")
joblib.dump(le, "label_encoder.pkl")
print("✅ Model trained and saved.")