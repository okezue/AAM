FROM python:3.11-slim
WORKDIR /workspace
COPY . /workspace
RUN pip install --no-cache-dir -e .
ENTRYPOINT ["python", "-m", "aamemory.cli"]
CMD ["smoke", "--config", "configs/experiments/l0_cpu_smoke.yaml"]
