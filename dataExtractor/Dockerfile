# Dockerfile

# 1) Use an official slim Python image
FROM python:3.9-slim

# 2) Set working directory
WORKDIR /app

# 4) Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# 5) Copy your application code
COPY . .

# 6) Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# 7) Expose port
EXPOSE 8080

# 8) Default command
CMD ["python", "app.py"]
