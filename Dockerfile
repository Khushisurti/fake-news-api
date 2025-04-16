# Use Python base image
FROM python:3.9

# Create app directory
WORKDIR /app

# Copy files
COPY . .

# Install Python dependencies
RUN pip install -r requirements.txt

# Expose port
EXPOSE 10000

# Run Flask app
CMD ["python", "app.py"]
