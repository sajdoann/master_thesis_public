# How to Start Apache Tika Server

The Python `tika` library requires a Java-based Tika server to be running. The easiest way is to use Docker.

**Important for OCR**: The `docker-compose.yml` file uses the `apache/tika:full` image which includes Tesseract OCR support. This is required for processing image-based PDFs. OCR is enabled by default in the comparison script.

## Quick Start with Docker (Recommended)

### Option 1: Using Docker Compose

```bash
cd pdf_extractor
docker compose up
```

This uses the `apache/tika:full` image which includes Tesseract OCR support.

Or use the helper script:
```bash
chmod +x start_tika_docker.sh
./start_tika_docker.sh
```

### Option 2: Simple Docker Command

For OCR support (recommended):
```bash
docker run -d \
  --name tika-server \
  -p 9998:9998 \
  apache/tika:full \
  java -jar /tika-server-standard.jar --host=0.0.0.0 --port=9998
```

For basic text extraction without OCR:
```bash
docker run -d \
  --name tika-server \
  -p 9998:9998 \
  apache/tika:latest \
  java -jar /tika-server-standard.jar --host=0.0.0.0 --port=9998
```

Or use the helper script:
```bash
chmod +x start_tika_docker_simple.sh
./start_tika_docker_simple.sh
```

The server will be available at `http://localhost:9998`

### Stop the Server

```bash
# If using docker-compose
docker compose down

# If using simple docker command
docker stop tika-server
docker rm tika-server
```

## Manual Java Installation (Alternative)

If you prefer not to use Docker:

1. **Install Java (JDK/JRE)**
   ```bash
   # On Ubuntu/Debian
   sudo apt update
   sudo apt install default-jdk default-jre
   
   # Verify installation
   java --version
   ```

2. **Download Tika Server JAR**
   - Visit: https://tika.apache.org/download.html
   - Download the latest `tika-server-<version>.jar` file

3. **Start the server**
   ```bash
   java -jar tika-server-<version>.jar
   ```

## Using the Server

Once the server is running (via Docker or manually), the Python script will automatically connect to it at `http://localhost:9998`.

### OCR Configuration

By default, OCR is **enabled** for Tika extractor. This allows processing of image-based PDFs. The script uses the `ocr_and_text_extraction` strategy, which:
- First attempts to extract text from the PDF's text layer
- Falls back to OCR if no text layer is available (image-based PDFs)

To disable OCR:
```bash
python compare_extractors.py --disable-tika-ocr
```

### Custom Endpoint

If you need to use a different endpoint, you can set the `TIKA_SERVER_ENDPOINT` environment variable:

```bash
export TIKA_SERVER_ENDPOINT=http://localhost:9998
python compare_extractors.py
```

## Troubleshooting

### Docker not installed
- Install Docker: https://docs.docker.com/get-docker/

### Port 9998 already in use
- Stop the existing service: `docker stop tika-server`
- Or use a different port and update the endpoint accordingly

### Connection refused
- Verify the server is running: `curl http://localhost:9998/tika`
- Check Docker container: `docker ps | grep tika`
- View logs: `docker logs tika-server`

### OCR not working
- Ensure you're using the `apache/tika:full` image (includes Tesseract OCR)
- Check that OCR is enabled: the script prints "Tika OCR is ENABLED" at startup
- OCR processing takes longer than text extraction - be patient with image-based PDFs
- If OCR fails, try disabling it with `--disable-tika-ocr` to test text extraction only

