FROM python:3.11-slim

WORKDIR /app

# Serving dependencies only. The training stack (torch, scikit-learn,
# matplotlib) is not on the request path and would add roughly 2 GB.
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src/ src/
COPY api/ api/

# Precomputed neighbours and the title lookup, about 30 MB. No ratings or
# training data are needed to serve.
COPY data/processed/item_neighbors.npz data/processed/
COPY data/raw/ml-32m/movies.csv data/raw/ml-32m/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# 0.0.0.0, not localhost: binding to the loopback inside a container makes
# the service unreachable from outside it.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
