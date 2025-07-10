# predict.py

import joblib

model = joblib.load("model.pkl")
le = joblib.load("label_encoder.pkl")

sample = [[5.5, 2.1]]
prediction = model.predict(sample)
print("Prediction:", le.inverse_transform(prediction)[0])