# ml_api/views.py

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import joblib
import os
from django.conf import settings

model_path = os.path.join(settings.BASE_DIR, "ml_api", "model.pkl")
encoder_path = os.path.join(settings.BASE_DIR, "ml_api", "label_encoder.pkl")

model = joblib.load("model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

class PredictView(APIView):
    def post(self, request):
        try:
            engine_size = float(request.data.get("engine_size"))
            horsepower = float(request.data.get("horsepower"))

            prediction = model.predict([[engine_size, horsepower]])
            label = label_encoder.inverse_transform(prediction)[0]

            return Response({"prediction": label})
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)