"""
Lightweight LlamaCloud MCP Server
==================================
Queries LlamaCloud indexes via the REST API directly,
without the heavy llama-index-core / llama-cloud SDK dependencies.

This makes uvx startup near-instant (3 deps vs 50+).
"""

import click
import os
import json
import logging

import httpx
from mcp.server.fastmcp import Context, FastMCP
from typing import Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

LLAMACLOUD_API_BASE = "https://api.cloud.llamaindex.ai/api/v1"

mcp = FastMCP("llamacloud-lite")


async def _get_pipeline_id(
    client: httpx.AsyncClient,
    index_name: str,
    project_id: Optional[str],
    org_id: Optional[str],
) -> str:
    """Resolve an index name to a pipeline ID via the LlamaCloud API."""
    params = {}
    if project_id:
        params["project_id"] = project_id
    if org_id:
        params["organization_id"] = org_id

    # List pipelines and find the one matching our index name
    resp = await client.get(f"{LLAMACLOUD_API_BASE}/pipelines", params=params)
    resp.raise_for_status()
    pipelines = resp.json()

    for pipeline in pipelines:
        if pipeline.get("name") == index_name:
            return pipeline["id"]

    raise ValueError(
        f"Index/pipeline '{index_name}' not found. "
        f"Available: {[p.get('name') for p in pipelines]}"
    )


async def _retrieve(
    client: httpx.AsyncClient,
    pipeline_id: str,
    query: str,
    top_k: int = 8,
) -> list[dict]:
    """Run a retrieval query against a LlamaCloud pipeline."""
    resp = await client.post(
        f"{LLAMACLOUD_API_BASE}/pipelines/{pipeline_id}/retrieve",
        json={
            "query": query,
            "search_params": {
                "dense_similarity_top_k": top_k,
            },
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.json()


def make_index_tool(
    index_name: str,
    api_key: str,
    project_id: Optional[str],
    org_id: Optional[str],
    top_k: int = 8,
) -> Callable[[Context, str], Awaitable[str]]:
    """Create a tool function that queries a specific LlamaCloud index."""

    # Cache the pipeline ID after first lookup
    _pipeline_id_cache: dict[str, str] = {}

    async def tool(ctx: Context, query: str) -> str:
        """Query the LlamaCloud index and return relevant results."""
        try:
            await ctx.info(f"Querying index '{index_name}' with: {query}")

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }

            async with httpx.AsyncClient(headers=headers) as client:
                # Resolve pipeline ID (cached after first call)
                if index_name not in _pipeline_id_cache:
                    pid = await _get_pipeline_id(client, index_name, project_id, org_id)
                    _pipeline_id_cache[index_name] = pid

                pipeline_id = _pipeline_id_cache[index_name]

                # Run the retrieval
                results = await _retrieve(client, pipeline_id, query, top_k)

            # Format results for the LLM
            if not results:
                return f"No results found for query: {query}"

            # Handle both list-of-nodes and dict response formats
            nodes = results if isinstance(results, list) else results.get("retrieval_nodes", results.get("nodes", []))

            formatted = []
            for i, node in enumerate(nodes, 1):
                # Handle different response structures
                if isinstance(node, dict):
                    text = node.get("text", node.get("node", {}).get("text", ""))
                    score = node.get("score", node.get("similarity", "N/A"))
                    metadata = node.get("metadata", node.get("node", {}).get("metadata", {}))
                else:
                    text = str(node)
                    score = "N/A"
                    metadata = {}

                source = metadata.get("file_name", metadata.get("source", "unknown"))
                formatted.append(
                    f"--- Result {i} (score: {score}, source: {source}) ---\n{text}"
                )

            return "\n\n".join(formatted)

        except httpx.HTTPStatusError as e:
            error_body = e.response.text if e.response else "No response body"
            error_msg = f"LlamaCloud API error ({e.response.status_code}): {error_body}"
            await ctx.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Error querying index '{index_name}': {str(e)}"
            await ctx.error(error_msg)
            return error_msg

    return tool


@click.command()
@click.option(
    "--index",
    "indexes",
    multiple=True,
    required=False,
    type=str,
    help="Index definition in format name:description. Can be repeated.",
)
@click.option(
    "--project-id",
    required=False,
    type=str,
    help="LlamaCloud Project ID",
)
@click.option(
    "--org-id",
    required=False,
    type=str,
    help="LlamaCloud Organization ID",
)
@click.option(
    "--transport",
    default="stdio",
    type=click.Choice(["stdio", "sse", "streamable-http"]),
    help="MCP transport mode",
)
@click.option(
    "--api-key",
    required=False,
    type=str,
    help="LlamaCloud API key (or set LLAMA_CLOUD_API_KEY env var)",
)
@click.option(
    "--top-k",
    default=8,
    type=int,
    help="Number of results to retrieve per query (default: 8)",
)
@click.option(
    "--port",
    default=8000,
    type=int,
    help="Port for HTTP transports (default: 8000)",
)
@click.option(
    "--host",
    default="0.0.0.0",
    type=str,
    help="Host for HTTP transports (default: 0.0.0.0)",
)
def main(
    indexes: Optional[list[str]],
    project_id: Optional[str],
    org_id: Optional[str],
    transport: str,
    api_key: Optional[str],
    top_k: int,
    port: int,
    host: str,
) -> None:
    """Lightweight LlamaCloud MCP Server - queries indexes via REST API."""

    api_key = api_key or os.getenv("LLAMA_CLOUD_API_KEY")
    if not api_key:
        raise click.BadParameter(
            "API key required. Use --api-key or set LLAMA_CLOUD_API_KEY env var."
        )

    # Parse indexes
    index_info = []
    if indexes:
        for idx in indexes:
            if ":" not in idx:
                raise click.BadParameter(
                    f"Index '{idx}' must be in format name:description"
                )
            name, description = idx.split(":", 1)
            index_info.append((name.strip(), description.strip()))

    if not index_info:
        raise click.BadParameter("At least one --index is required.")

    # Register a tool for each index
    for name, description in index_info:
        tool_func = make_index_tool(name, api_key, project_id, org_id, top_k)
        mcp.tool(name=f"query_{name}", description=description)(tool_func)

    # Run the server
    if transport == "stdio":
        mcp.run(transport="stdio")
    elif transport == "sse":
        mcp.run(transport="sse")
    elif transport == "streamable-http":
        mcp.run(transport="streamable-http", host=host, port=port)


if __name__ == "__main__":
    main()
