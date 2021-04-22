# syntax=docker/dockerfile:1
# FROM tiangolo/uwsgi-nginx-flask:latest
FROM tensorflow/tensorflow

WORKDIR /app

RUN apt update && \
apt install --no-install-recommends -y build-essential gcc curl ca-certificates python3 ffmpeg xvfb libsm6 libxext6 && \
apt clean && rm -rf /var/lib/apt/lists/*

COPY . .

ENV PATH="$PATH:/root/.local/bin"

# ENV DISPLAY=""

# ENV XAUTHORITY=""
# ENV FLASK_APP=serve_ml.py

# ENV FLASK_RUN_HOST=0.0.0.0
RUN pip install --no-cache-dir --user -r requirements.txt

RUN pip install --upgrade --force-reinstall numpy

RUN pip install PyVirtualDisplay

# EXPOSE 8080

CMD ["python3", "-m" , "flask", "run", "--host=0.0.0.0"]