import os
import httpx
from fastmcp import FastMCP, Context
from fastmcp.server.dependencies import get_http_headers

# Point this at your platform
PLATFORM_API_BASE = os.environ.get("PLATFORM_API_BASE", "https://api.mydomain.com")

mcp = FastMCP(name="MemoryServer", instructions="""
Public remote MCP server for simple memory ops. 
Supply your platform API key as a header: 
- X-Platform-API-Key: <key>  (preferred) 
or 
- Authorization: Bearer <key>
""")

def _extract_api_key() -> str | None:
    # Works for HTTP transport: read current request headers
    # (FastMCP provides helpers for this)
    # https://gofastmcp.com/servers/context  (HTTP Headers & Access Tokens)
    headers = get_http_headers()
    if not headers:
        return None
    h = headers.get("x-platform-api-key") or headers.get("X-Platform-API-Key")
    if h:
        return h.strip()
    auth = headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None

@mcp.tool
async def memorize(key: str, value: str, ttl_seconds: int | None = None, ctx: Context | None = None) -> dict:
    """Store a value under a key in our platform."""
    api_key = _extract_api_key()
    if not api_key:
        raise ValueError("Missing API key. Pass X-Platform-API-Key or Authorization: Bearer <key>.")
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            f"{PLATFORM_API_BASE}/v1/memory",
            json={"key": key, "value": value, "ttl_seconds": ttl_seconds},
            headers={"Authorization": f"Bearer {api_key}"},
        )
        resp.raise_for_status()
        return resp.json()

@mcp.tool
async def recall(key: str, ctx: Context | None = None) -> dict:
    """Retrieve a value previously stored under a key in our platform."""
    api_key = _extract_api_key()
    if not api_key:
        raise ValueError("Missing API key. Pass X-Platform-API-Key or Authorization: Bearer <key>.")
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            f"{PLATFORM_API_BASE}/v1/memory/{key}",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        resp.raise_for_status()
        return resp.json()

@mcp.tool
def add(a: int, b: int) -> int:
    """Simple math example tool."""
    return a + b

if __name__ == "__main__":
    # For local testing only (remote hosting uses Cloud)
    mcp.run(transport="http", host="127.0.0.1", port=8000)
