FROM python:3.11-slim

WORKDIR /app

# Install CPU-only PyTorch explicitly
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code and the trained model
COPY . .

# Expose the port
ENV PORT 8000
EXPOSE 8000

# Run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]