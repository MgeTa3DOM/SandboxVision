# Multi-stage build for SandboxVision
FROM python:3.10-slim as builder

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.10-slim

WORKDIR /app

COPY --from=builder /root/.local /root/.local

ENV PATH=/root/.local/bin:$PATH

COPY src/ ./src/
COPY setup.py .

RUN pip install --no-cache-dir -e .

EXPOSE 8765

# Note: This CMD assumes you will create a main entry point in your package
# For now, we'll need a runnable script. Let's assume a main.py in src
# We will need to create a main entrypoint later.
CMD ["sandboxvision"]