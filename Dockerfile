FROM python:3.10-alpine

# Install dependencies
RUN apk add --no-cache gcc libffi-dev musl-dev

# Set working directory
WORKDIR /app

# Copy application code
COPY . /app

# Install python dependencies (remove cache to save space)
RUN pip install --no-cache-dir -r requirements.txt

# Clean up unnecessary files
RUN rm -rf /var/lib/apt/lists/* /app/tests /app/.git

# Command to run the app
CMD ["python", "app.py"]
