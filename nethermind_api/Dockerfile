FROM python:3.11-slim

WORKDIR /app

# Install curl (or wget) which is needed by the install script
RUN apt-get update && \
    apt-get install -y curl gcc build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Download and make install script executable
ADD --chmod=755 https://astral.sh/uv/install.sh /install.sh

# Run the install script so that `uv` becomes available
RUN /install.sh

# Now it's safe to run `uv` since it's installed
RUN /root/.local/bin/uv pip install --system -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "app.py"]
