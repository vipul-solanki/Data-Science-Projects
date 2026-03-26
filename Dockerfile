# Use Python image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy all files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Flask port
EXPOSE 5000

# Run app
CMD ["python", "-m", "src.api.app"]