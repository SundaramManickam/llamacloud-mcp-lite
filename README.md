# llamacloud-mcp-lite

Lightweight MCP server for LlamaCloud indexes. Replaces the heavy `llamacloud-mcp` package (which depends on `llama-index-core` and takes minutes to install) with direct REST API calls. Installs in seconds.

## Why this exists

The official `llamacloud-mcp` crashes Claude Desktop because it pulls in ~50 heavy Python packages via `uvx`, causing timeouts during the MCP handshake. This version has only 3 dependencies: `mcp`, `httpx`, and `click`.

## Quick Start (Local - requires Python)

```bash
uvx llamacloud-mcp-lite \
  --api-key YOUR_LLAMA_CLOUD_API_KEY \
  --org-id YOUR_ORG_ID \
  --project-id YOUR_PROJECT_ID \
  --index "my-index:Description of what this index contains"
```

## Remote Deployment (for sharing - no Python needed on clients)

Deploy this server with `--transport streamable-http` on any cloud provider, then clients connect via URL in their MCP config.

### Deploy to Railway (free tier available)

1. Push this repo to GitHub
2. Connect to Railway.app
3. Set environment variables: `LLAMA_CLOUD_API_KEY`, `LLAMA_CLOUD_ORG_ID`, `LLAMA_CLOUD_PROJECT_ID`, and `INDEX_1`, `INDEX_2`, `INDEX_3` (format: `name:description`)
4. Deploy

### Deploy with Docker

```bash
docker build -f deploy/Dockerfile -t llamacloud-mcp-lite .
docker run -p 8000:8000 llamacloud-mcp-lite \
  --transport streamable-http --port 8000 \
  --api-key YOUR_KEY --org-id YOUR_ORG --project-id YOUR_PROJECT \
  --index "neca-manual:NECA labor hours and material costs"
```

## Claude Desktop / Cowork Plugin Config

### Local (stdio) - for machines with Python/uvx
```json
{
  "mcpServers": {
    "llamacloud": {
      "command": "uvx",
      "args": [
        "llamacloud-mcp-lite",
        "--api-key", "YOUR_KEY",
        "--org-id", "YOUR_ORG_ID",
        "--project-id", "YOUR_PROJECT_ID",
        "--index", "index-name:Description"
      ]
    }
  }
}
```

### Remote (streamable-http) - for machines with only Claude installed
```json
{
  "mcpServers": {
    "llamacloud": {
      "url": "https://your-deployed-server.railway.app/mcp",
      "transport": "streamable-http"
    }
  }
}
```

## Per-Client Index Scoping

When deployed with `--transport streamable-http`, you can restrict which indexes a client can access by adding `?indexes=` to the URL. The server is configured with all indexes at startup, but each client only sees the ones in their query param.

```json
{
  "mcpServers": {
    "llamacloud": {
      "url": "https://your-server.railway.app/mcp?indexes=neca-manual,pricing-data",
      "transport": "streamable-http"
    }
  }
}
```

- `/mcp?indexes=foo,bar` — client only sees `query_foo` and `query_bar` tools
- `/mcp?indexes=foo` — client only sees `query_foo`
- `/mcp` — client sees all configured indexes (no filtering)

Unknown index names return a 400 error.

## License

MIT
