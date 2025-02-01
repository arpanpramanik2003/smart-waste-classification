# Use the official Python image
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy the project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8501 (Streamlit default port)
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "trashnetST.py", "--server.port=8080", "--server.address=0.0.0.0"]
