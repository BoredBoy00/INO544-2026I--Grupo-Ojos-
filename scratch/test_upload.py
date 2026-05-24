import sys
import os
import io

# Agregar directorio raíz al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app

client = app.test_client()

with open("Ojos/img_transformada_0.jpg", "rb") as f:
    img_bytes = f.read()

data = {
    "file": (io.BytesIO(img_bytes), "img_transformada_0.jpg"),
    "threshold": "0.5",
    "low_threshold": "0.001"
}

response = client.post("/detect", data=data, content_type="multipart/form-data")
print("Status Code:", response.status_code)
print("Response JSON:", response.get_json())
