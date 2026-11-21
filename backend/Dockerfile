FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
ENV ACOUSTICASE_DATA_DIR=/app/data MPLCONFIGDIR=/app/data/matplotlib
EXPOSE 8000
CMD ["uvicorn", "acousticase.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
