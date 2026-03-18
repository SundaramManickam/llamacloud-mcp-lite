"""
Lightweight LlamaCloud MCP Server
==================================
Queries LlamaCloud indexes via the REST API directly,
without the heavy llama-index-core / llama-cloud SDK dependencies.

This makes uvx startup near-instant (3 deps vs 50+).

Supports two modes for per-connector index isolation:

1. Path-based routing (recommended for Claude connectors):
     /indexes/Material-pricing-all/mcp  -> only sees Material-pricing-all
     /indexes/Inventory/mcp             -> only sees Inventory
     /mcp                               -> sees all indexes

2. Query-param scoping (for clients that support query params):
     /mcp?indexes=foo,bar&x-api-key=SECRET
"""

import click
import contextlib
import contextvars
import os
import logging
from contextlib import AsyncExitStack
from urllib.parse import parse_qs

import httpx
import uvicorn
from mcp.server.fastmcp import Context, FastMCP
from mcp.types import Tool as MCPTool
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.types import ASGIApp, Receive, Scope, Send
from typing import Any, Awaitable, Callable, Optional, Sequence

logger = logging.getLogger(__name__)

LLAMACLOUD_API_BASE = "https://api.cloud.llamaindex.ai/api/v1"

_pipeline_id_cache: dict[str, str] = {}

# Set by the ASGI middleware before each request hits the MCP handler.
# None = no filtering (all tools visible).
_allowed_tools_var: contextvars.ContextVar[Optional[set[str]]] = contextvars.ContextVar(
    "_allowed_tools_var", default=None
)

def _fetch_all_indexes(
    api_key: str,
    project_id: Optional[str] = None,
    org_id: Optional[str] = None,
) -> list[tuple[str, str]]:
    """
    Fetch all indexes (pipelines) from the LlamaCloud API synchronously at startup.
    Returns a list of (name, description) tuples.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    params: dict[str, str] = {}
    if project_id:
        params["project_id"] = project_id
    if org_id:
        params["organization_id"] = org_id

    with httpx.Client(headers=headers) as client:
        resp = client.get(f"{LLAMACLOUD_API_BASE}/pipelines", params=params)
        resp.raise_for_status()
        pipelines = resp.json()

    results = []
    for p in pipelines:
        name = p.get("name", "")
        if not name:
            continue
        # Use the pipeline's description if available, otherwise generate one
        desc = p.get("description") or f"Query the '{name}' index"
        # Cache the pipeline ID while we're at it
        if p.get("id"):
            _pipeline_id_cache[name] = p["id"]
        results.append((name, desc))

    return results


async def _get_pipeline_id(
    client: httpx.AsyncClient,
    index_name: str,
    project_id: Optional[str],
    org_id: Optional[str],
) -> str:
    """Resolve an index name to a pipeline ID via the LlamaCloud API."""
    if index_name in _pipeline_id_cache:
        return _pipeline_id_cache[index_name]

    params = {}
    if project_id:
        params["project_id"] = project_id
    if org_id:
        params["organization_id"] = org_id

    resp = await client.get(f"{LLAMACLOUD_API_BASE}/pipelines", params=params)
    resp.raise_for_status()
    pipelines = resp.json()

    for pipeline in pipelines:
        if pipeline.get("name") == index_name:
            _pipeline_id_cache[index_name] = pipeline["id"]
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
                pipeline_id = await _get_pipeline_id(
                    client, index_name, project_id, org_id
                )
                results = await _retrieve(client, pipeline_id, query, top_k)

            if not results:
                return f"No results found for query: {query}"

            nodes = (
                results
                if isinstance(results, list)
                else results.get(
                    "retrieval_nodes", results.get("nodes", [])
                )
            )

            formatted = []
            for i, node in enumerate(nodes, 1):
                if isinstance(node, dict):
                    text = node.get(
                        "text", node.get("node", {}).get("text", "")
                    )
                    score = node.get("score", node.get("similarity", "N/A"))
                    metadata = node.get(
                        "metadata", node.get("node", {}).get("metadata", {})
                    )
                else:
                    text = str(node)
                    score = "N/A"
                    metadata = {}

                source = metadata.get(
                    "file_name", metadata.get("source", "unknown")
                )
                formatted.append(
                    f"--- Result {i} (score: {score}, source: {source}) ---\n{text}"
                )

            return "\n\n".join(formatted)

        except httpx.HTTPStatusError as e:
            error_body = (
                e.response.text if e.response else "No response body"
            )
            error_msg = f"LlamaCloud API error ({e.response.status_code}): {error_body}"
            await ctx.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Error querying index '{index_name}': {str(e)}"
            await ctx.error(error_msg)
            return error_msg

    return tool


class ScopedFastMCP(FastMCP):
    """
    FastMCP subclass that filters tools based on a per-request ContextVar.

    When _allowed_tools_var is set, list_tools only returns matching tools
    and call_tool rejects anything outside the set.
    """

    async def list_tools(self) -> list[MCPTool]:
        all_tools = await super().list_tools()
        allowed = _allowed_tools_var.get()
        if allowed is None:
            return all_tools
        return [t for t in all_tools if t.name in allowed]

    async def call_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> Sequence[Any] | dict[str, Any]:
        allowed = _allowed_tools_var.get()
        if allowed is not None and name not in allowed:
            raise ValueError(
                f"Tool '{name}' is not available for this connection."
            )
        return await super().call_tool(name, arguments)


class ApiKeyAuthMiddleware:
    """
    ASGI middleware that checks ?x-api-key= against the server's configured key.
    If no key is configured on the server, all requests pass through.
    """

    def __init__(self, app: ASGIApp, expected_key: Optional[str]):
        self.app = app
        self.expected_key = expected_key

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not self.expected_key:
            await self.app(scope, receive, send)
            return

        query_string = scope.get("query_string", b"").decode()
        qs = parse_qs(query_string)
        provided_keys = qs.get("x-api-key", [])
        provided_key = provided_keys[0] if provided_keys else None

        if not provided_key or provided_key != self.expected_key:
            await send({
                "type": "http.response.start",
                "status": 401,
                "headers": [[b"content-type", b"text/plain"]],
            })
            await send({
                "type": "http.response.body",
                "body": b"Unauthorized: invalid or missing x-api-key",
            })
            return

        await self.app(scope, receive, send)


class IndexScopingMiddleware:
    """
    ASGI middleware that reads ?indexes=foo,bar from the query string
    and sets the _allowed_tools_var ContextVar before the request
    reaches the MCP handler.
    """

    def __init__(self, app: ASGIApp, all_index_names: set[str]):
        self.app = app
        self.all_index_names = all_index_names

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        query_string = scope.get("query_string", b"").decode()
        qs = parse_qs(query_string)
        requested = qs.get("indexes", [])

        if not requested:
            _allowed_tools_var.set(None)
            await self.app(scope, receive, send)
            return

        allowed_indexes: set[str] = set()
        for val in requested:
            allowed_indexes.update(
                name.strip() for name in val.split(",") if name.strip()
            )

        bad = allowed_indexes - self.all_index_names
        if bad:
            body = (
                f"Unknown indexes: {', '.join(sorted(bad))}. "
                f"Available: {', '.join(sorted(self.all_index_names))}"
            ).encode()
            await send({
                "type": "http.response.start",
                "status": 400,
                "headers": [[b"content-type", b"text/plain"]],
            })
            await send({"type": "http.response.body", "body": body})
            return

        allowed_tools = {f"query_{n}" for n in allowed_indexes}
        _allowed_tools_var.set(allowed_tools)

        scope_clean = dict(scope)
        scope_clean["query_string"] = b""
        await self.app(scope_clean, receive, send)


def _build_mcp(
    index_info: list[tuple[str, str]],
    api_key: str,
    project_id: Optional[str],
    org_id: Optional[str],
    top_k: int,
    stateless: bool = False,
    remote: bool = False,
) -> FastMCP:
    """Build a FastMCP instance with the given indexes registered as tools."""
    cls = ScopedFastMCP if stateless else FastMCP
    kwargs: dict[str, Any] = {"stateless_http": stateless}
    if remote:
        # Disable DNS rebinding protection for remote deployments (e.g. Render).
        # Without this, the MCP SDK rejects any Host header that isn't localhost.
        kwargs["host"] = "0.0.0.0"
        kwargs["transport_security"] = TransportSecuritySettings(
            enable_dns_rebinding_protection=False,
        )
    server = cls("llamacloud-lite", **kwargs)
    for name, description in index_info:
        tool_func = make_index_tool(name, api_key, project_id, org_id, top_k)
        server.tool(name=f"query_{name}", description=description)(tool_func)
    return server


def _build_scoped_http_app(
    all_indexes: dict[str, str],
    api_key: str,
    project_id: Optional[str],
    org_id: Optional[str],
    top_k: int,
    x_api_key: Optional[str] = None,
) -> ASGIApp:
    """
    Build a Starlette app with:

    1. Per-index path-based routes (for Claude connectors):
         /indexes/<index-name>/mcp  -> isolated MCP server with only that index

    2. Combined route with query-param scoping (backward compat):
         /mcp                       -> all indexes (filterable via ?indexes=)

    Middleware order on the combined route (outermost first):
      ApiKeyAuthMiddleware  -> checks ?x-api-key=
      IndexScopingMiddleware -> filters tools by ?indexes=
      MCP Starlette app     -> handles the actual MCP protocol
    """
    # --- Per-index MCP servers for path-based routing ---
    per_index_servers: list[FastMCP] = []
    routes = []
    for name, description in all_indexes.items():
        server = _build_mcp(
            [(name, description)], api_key, project_id, org_id, top_k,
            stateless=True, remote=True,
        )
        per_index_servers.append(server)
        per_index_app: ASGIApp = server.streamable_http_app()
        if x_api_key:
            per_index_app = ApiKeyAuthMiddleware(per_index_app, x_api_key)
        routes.append(Mount(f"/indexes/{name}", app=per_index_app))

    # --- Combined MCP server with all indexes (query-param scoping) ---
    index_info = list(all_indexes.items())
    combined_server = _build_mcp(
        index_info, api_key, project_id, org_id, top_k,
        stateless=True, remote=True,
    )
    per_index_servers.append(combined_server)
    combined_app: ASGIApp = combined_server.streamable_http_app()
    combined_app = IndexScopingMiddleware(combined_app, set(all_indexes.keys()))
    combined_app = ApiKeyAuthMiddleware(combined_app, x_api_key)
    routes.append(Mount("/", app=combined_app))

    # --- Combined lifespan to initialize all session managers ---
    @contextlib.asynccontextmanager
    async def combined_lifespan(app):
        async with AsyncExitStack() as stack:
            for srv in per_index_servers:
                await stack.enter_async_context(srv.session_manager.run())
            yield

    return Starlette(routes=routes, lifespan=combined_lifespan)


@click.command()
@click.option(
    "--index",
    "indexes",
    multiple=True,
    required=False,
    type=str,
    help="Index definition in format name:description. Can be repeated. If omitted, all indexes are auto-discovered from the LlamaCloud API.",
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
@click.option(
    "--x-api-key",
    "x_api_key",
    required=False,
    type=str,
    help="API key clients must pass as ?x-api-key= query param (or set X_API_KEY env var)",
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
    x_api_key: Optional[str],
) -> None:
    """Lightweight LlamaCloud MCP Server - queries indexes via REST API."""

    api_key = api_key or os.getenv("LLAMA_CLOUD_API_KEY")
    if not api_key:
        raise click.BadParameter(
            "API key required. Use --api-key or set LLAMA_CLOUD_API_KEY env var."
        )

    x_api_key = x_api_key or os.getenv("X_API_KEY")
    project_id = project_id or os.getenv("LLAMA_CLOUD_PROJECT_ID")
    org_id = org_id or os.getenv("LLAMA_CLOUD_ORG_ID")

    index_info: list[tuple[str, str]] = []
    if indexes:
        for idx in indexes:
            if ":" not in idx:
                raise click.BadParameter(
                    f"Index '{idx}' must be in format name:description"
                )
            name, description = idx.split(":", 1)
            index_info.append((name.strip(), description.strip()))

    if not index_info:
        # Auto-discover all indexes from the LlamaCloud API
        logger.info("No --index flags provided, auto-discovering from LlamaCloud API...")
        index_info = _fetch_all_indexes(api_key, project_id, org_id)
        if not index_info:
            raise click.BadParameter(
                "No indexes found via LlamaCloud API. "
                "Provide --index flags or check your API key / project-id / org-id."
            )
        logger.info(f"Discovered {len(index_info)} indexes: {[n for n, _ in index_info]}")

    if transport == "streamable-http":
        all_indexes = {name: desc for name, desc in index_info}
        app = _build_scoped_http_app(
            all_indexes, api_key, project_id, org_id, top_k,
            x_api_key=x_api_key,
        )
        uvicorn.run(app, host=host, port=port)
    else:
        mcp = _build_mcp(index_info, api_key, project_id, org_id, top_k)
        if transport == "stdio":
            mcp.run(transport="stdio")
        elif transport == "sse":
            mcp.run(transport="sse")


if __name__ == "__main__":
    main()