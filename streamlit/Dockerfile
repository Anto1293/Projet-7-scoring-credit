# streamlit/Dockerfile

FROM python:3.10-slim

WORKDIR /streamlit

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY streamlit_final.py .

# Expose le port de Streamlit (8501 par défaut)
EXPOSE 8501

CMD ["streamlit", "run", "streamlit_final.py", "--server.port=8501", "--server.address=0.0.0.0"]
