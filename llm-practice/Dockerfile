FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/jovyan/.cache/huggingface \
    PYTHONPATH=/home/jovyan/work/src

RUN apt-get update && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*

# non-root user; the cache dir is created here so the named volume inherits ownership
RUN useradd -m -u 1000 jovyan \
    && mkdir -p /home/jovyan/.cache/huggingface /home/jovyan/work \
    && chown -R jovyan:jovyan /home/jovyan

WORKDIR /home/jovyan/work
COPY requirements.txt .
RUN pip install -r requirements.txt

USER jovyan
EXPOSE 8888
# JUPYTER_TOKEN env var is read automatically by Jupyter Server
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser"]
