FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . .

# Vérifier que TensorFlow est importable
RUN python -c "import tensorflow as tf; print(tf.__version__)"

CMD ["python", "test.py"]