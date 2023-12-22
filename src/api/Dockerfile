# usage: docker build -f Dockerfile -t hiseulgi/medical-leaf-api:latest .
FROM python:3.11.5-slim

# Install dependencies
RUN DEBIAN_FRONTEND=noninteractive \
    apt-get update && apt-get install -y --no-install-recommends \
    wget curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /tmp/requirements.txt
RUN rm /tmp/requirements.txt

# set working directory
WORKDIR /app

# entrypoint
CMD [ "bash" ]