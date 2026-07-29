FROM python:3.12-slim

WORKDIR /app

# CPU-only torch, installed first and deliberately from PyTorch's cpu index.
# The default PyPI wheel bundles ~3GB of CUDA libraries; nothing here uses a GPU
# (faiss-cpu, MiniLM inference over 4k chunks), so they are dead weight in the
# image. Installing it up front means the requirements install below finds the
# torch constraint already satisfied and will not pull the CUDA build.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

# Deps before code so source edits don't invalidate the dependency layer.
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
