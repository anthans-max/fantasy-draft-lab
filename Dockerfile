FROM python:3.11-slim

WORKDIR /app

# System deps (optional but helpful for many python wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

EXPOSE 8000

# Streamlit settings for container environments
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

CMD ["/bin/sh","-c","streamlit run ui/app.py --server.address=0.0.0.0 --server.port=${WEBSITES_PORT:-${PORT:-8000}} --server.headless=true"]
