# FarmGuard AI — Dockerfile
# Packages the entire application into a Docker image

# Start with Python 3.10 as our base
FROM python:3.12-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first (for faster rebuilds)
COPY requirements.txt .

# Install all Python packages
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Copy entire project into container
COPY . .

# Create temp folder for image uploads
RUN mkdir -p temp_uploads

# Expose ports for FastAPI and Streamlit
EXPOSE 8000
EXPOSE 8501

# Make the entrypoint script executable
RUN chmod +x entrypoint.sh

# Fixes JSONArgsRecommended warning and maps both processes cleanly
CMD ["./entrypoint.sh"]

