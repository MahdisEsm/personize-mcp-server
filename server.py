import os
import httpx
from fastmcp import FastMCP, Context
from fastmcp.server.dependencies import get_http_headers

# Point this at your platform
# PLATFORM_API_BASE = os.environ.get("PLATFORM_API_BASE", "https://api.mydomain.com")




mcp = FastMCP(name="Personize", instructions="""
Public remote MCP server for tools like research_company_with_tavily, memorizer and Recall.
Supply Personize API key as a header: 
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

# @mcp.tool
# async def memorize(key: str, value: str, ttl_seconds: int | None = None, ctx: Context | None = None) -> dict:
#     """Store a value under a key in our platform."""
#     api_key = _extract_api_key()
#     if not api_key:
#         raise ValueError("Missing API key. Pass X-Platform-API-Key or Authorization: Bearer <key>.")
#     async with httpx.AsyncClient(timeout=15) as client:
#         resp = await client.post(
#             f"{PLATFORM_API_BASE}/v1/memory",
#             json={"key": key, "value": value, "ttl_seconds": ttl_seconds},
#             headers={"Authorization": f"Bearer {api_key}"},
#         )
#         resp.raise_for_status()
#         return resp.json()

# @mcp.tool
# async def recall(key: str, ctx: Context | None = None) -> dict:
#     """Retrieve a value previously stored under a key in our platform."""
#     api_key = _extract_api_key()
#     if not api_key:
#         raise ValueError("Missing API key. Pass X-Platform-API-Key or Authorization: Bearer <key>.")
#     async with httpx.AsyncClient(timeout=15) as client:
#         resp = await client.get(
#             f"{PLATFORM_API_BASE}/v1/memory/{key}",
#             headers={"Authorization": f"Bearer {api_key}"},
#         )
#         resp.raise_for_status()
#         return resp.json()

@mcp.tool(description="Query a company's site via Tavily: Products, Services, Team, News, Key customers, FAQ, Pricing")
async def research_company_with_tavily(company_name: str, website_url: str, ctx: Context | None = None) -> dict:
    """
    Calls Tavily API to search within the given website for:
    Products, Services, Team, News, Key customers, FAQ, Pricing
    Returns Tavily's JSON response.
    """
    TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
    TAVILY_SEARCH_URL = "https://api.tavily.com/search"
    if not TAVILY_API_KEY:
        raise ValueError("Missing TAVILY_API_KEY environment variable on the server.")

    # Build the query exactly as requested
    query = (
        f"Products, Services, Team, News, Key customers, FAQ, Pricing of {company_name} "
        f"- site:{website_url}"
    )

    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        # Interpret “first 5 page” as top 5 results from Tavily
        "max_results": 5,
        # Use deeper crawling/quality for company research
        "search_depth": "advanced",
        # Keep the search restricted to the provided site as an extra guard
        "include_domains": [website_url],
        # Helpful structured answer + sources in the response
        "include_answer": True,
        # Keep payload small unless you specifically want full page text
        "include_raw_content": False,
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(TAVILY_SEARCH_URL, json=payload)
        # Raise for non-2xx so the model surfaces a clear error message
        resp.raise_for_status()
        data = resp.json()

    # Optional: log a small, non-sensitive breadcrumb
    if ctx:
        await ctx.info(f"Tavily search for '{company_name}' on {website_url} returned {len(data.get('results', []))} results.")

    return data



# if __name__ == "__main__":
#     # For local testing only (remote hosting uses Cloud)
#     mcp.run(transport="http", host="127.0.0.1", port=8000)
