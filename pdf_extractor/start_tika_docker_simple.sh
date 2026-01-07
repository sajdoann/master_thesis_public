#!/bin/bash
# Simple Docker command to start Tika server

echo "Starting Apache Tika server with Docker..."
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH"
    echo "Install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if container already exists and is running
if docker ps | grep -q tika-server; then
    echo "Tika server is already running!"
    echo "Access it at: http://localhost:9998"
    exit 0
fi

# Start the container
echo "Starting Tika server on http://localhost:9998"
echo "Press Ctrl+C to stop, or run 'docker stop tika-server' in another terminal"
echo ""

docker run -d \
  --name tika-server \
  -p 9998:9998 \
  --restart unless-stopped \
  apache/tika:latest \
  java -jar /tika-server-standard.jar --host=0.0.0.0 --port=9998

echo ""
echo "Tika server started!"
echo "Access it at: http://localhost:9998"
echo ""
echo "To stop: docker stop tika-server"
echo "To remove: docker rm tika-server"

