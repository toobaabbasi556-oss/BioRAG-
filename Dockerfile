FROM python:3.12-slim

WORKDIR /app

# Install deps first so code changes don't invalidate the dependency layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY eval/ ./eval/

# Corpus and FAISS index are mounted at run time — baking ~15MB of index plus
# a regenerable 3,000-abstract corpus into the image would make it awkward to
# share and stale the moment the corpus is rebuilt.
ENV PYTHONPATH=/app/src
EXPOSE 8501

CMD ["streamlit", "run", "src/biorag/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]
