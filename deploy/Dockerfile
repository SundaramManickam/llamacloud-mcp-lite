FROM python:3.12-slim

WORKDIR /app

# Copy the package
COPY pyproject.toml README.md ./
COPY llamacloud_mcp_lite/ ./llamacloud_mcp_lite/

# Install
RUN pip install --no-cache-dir .

# Expose port for streamable-http
EXPOSE 8000

# Default: run with streamable-http transport
# Override args via environment or docker run command
ENTRYPOINT ["llamacloud-mcp-lite"]
CMD ["--transport", "streamable-http", "--port", "8000"]
