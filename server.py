import os
import httpx
from fastmcp import FastMCP, Context
from fastmcp.server.dependencies import get_http_headers
import asyncio
from typing import Any, Dict, List, Optional

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



# ---- Pinecone search tool ----

@mcp.tool(description="Search Pinecone text records within a user's namespace.")
async def pinecone_search(UserId: str, query: str, top_k: int | None = None, ctx: Context | None = None) -> dict:
    """
    Inputs:
      - UserId: per-user namespace (string)
      - query: free-text query (string)
      - top_k: optional int; defaults to 5

    Returns:
      - success: bool
      - matches: list[str]              # text field(s) extracted from hits
      - raw: full Pinecone response     # pass-through for inspection
    """
    api_key = os.environ.get("PINECONE_API_KEY")
    index_name = os.environ.get("PINECONE_INDEX_NAME")
    index_host = os.environ.get("PINECONE_INDEX_HOST")  # optional
    if not api_key:
        raise ValueError("Missing PINECONE_API_KEY in environment.")
    if not index_name:
        raise ValueError("Missing PINECONE_INDEX_NAME in environment.")
    if not UserId or not UserId.strip():
        raise ValueError("UserId is required.")
    if not query or not query.strip():
        raise ValueError("query is required.")
    k = int(top_k or 5)

    try:
        from pinecone import Pinecone
    except Exception as e:
        raise RuntimeError("Pinecone SDK not available. Ensure 'pinecone' (v6/v7) is installed and redeploy.") from e

    # fields you want back (must match what you've stored)
    fields = ['text', 'Tag', 'Type', 'RecordId', 'UserId', 'TimeStamp']

    def _search_sync() -> Dict[str, Any]:
        pc = Pinecone(api_key=api_key)
        # v6/v7 Index constructor; host is optional if your SDK resolves it
        index = pc.Index(index_name, host=index_host) if index_host else pc.Index(index_name)

        # Records/text search on the INDEX (docs-preferred), namespaced via param
        # NOTE: top_k must be snake_case in Python
        result = index.search(
            namespace=UserId.strip(),
            query={"inputs": {"text": query}, "top_k": k},
            fields=fields
            # You can add rerank={...} here if desired
        )
        return result

    try:
        result = await asyncio.to_thread(_search_sync)
    except Exception as e:
        return {"success": False, "error": "pinecone_search_failed", "message": str(e)}

    # Normalize matches from result['result']['hits'][i]['fields']['text']
    matches: List[str] = []
    try:
        r = result if isinstance(result, dict) else getattr(result, "__dict__", {})
        hits = (r.get("result", {}) or {}).get("hits", [])
        for h in hits:
            f = h.get("fields", {})
            txt = f.get("text")
            if isinstance(txt, str):
                matches.append(txt)
    except Exception:
        pass

    if ctx:
        await ctx.info(f"Pinecone Records: {len(matches)} matches (namespace={UserId.strip()}, top_k={k}).")
    return {"success": True, "matches": matches, "raw": result}


# if __name__ == "__main__":
#     # For local testing only (remote hosting uses Cloud)
#     mcp.run(transport="http", host="127.0.0.1", port=8000)
