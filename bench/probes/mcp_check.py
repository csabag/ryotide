# /// script
# requires-python = ">=3.12"
# dependencies = ["mcp>=2.2,<3", "uvicorn>=0.30"]
# ///
"""End-to-end check of src/ryotide/mcp_server.py against a running decision API.

Starts the MCP server (Streamable HTTP with a random bearer token, then stdio), and checks:
401 without / with a wrong token, the tool list, every tool with a real decision, review fields,
that bad input comes back as a tool error, and 503 (not a pass) when forward-auth cannot reach the API. The token is generated here and never printed.

    uv run bench/probes/mcp_check.py [API_URL]        # default http://127.0.0.1:8778
"""
import asyncio
import json
import os
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.request

API = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8778").rstrip("/")
PORT = 8899
SERVER = os.path.join(os.path.dirname(__file__), "..", "..", "src", "ryotide", "mcp_server.py")
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


def status_of(url, headers):
    req = urllib.request.Request(url, method="POST", headers={**headers, "Content-Type": "application/json",
                                 "Accept": "application/json, text/event-stream"},
                                 data=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}).encode())
    try:
        return urllib.request.urlopen(req, timeout=10).status
    except urllib.error.HTTPError as e:
        return e.code


CONTEXT = ("Refund policy: purchases can be refunded within 30 days with a receipt. "
           "Opened software cannot be refunded. Customer: bought headphones 12 days ago, has the receipt.")


async def exercise(session, label):
    tools = sorted(t.name for t in (await session.list_tools()).tools)
    check(f"{label}: tool list", tools == ["decide_choice", "decide_many", "decide_score", "decide_yes_no", "engine_info"], str(tools))

    async def call(name, args):
        r = await session.call_tool(name, args)
        if r.is_error:
            return r, {"error": r.content[0].text if r.content else ""}
        data = r.structured_content if getattr(r, "structured_content", None) is not None else json.loads(r.content[0].text)
        return r, data

    r, d = await call("decide_yes_no", {"question": "Can the customer get a refund?", "context": CONTEXT, "review_threshold": 0.8})
    check(f"{label}: decide_yes_no", not r.is_error and d["answer"] in ("yes", "no") and 0 <= d["p_yes"] <= 1 and "needs_review" in d,
          f"answer={d.get('answer')} p_yes={d.get('p_yes', 0):.3f} review={d.get('needs_review')}")
    r, d = await call("decide_choice", {"question": "Which team should handle this?", "context": "My card was charged twice for one order.",
                                       "options": {"billing": "payments, charges, invoices", "tech": "bugs, login problems", "sales": "new purchases"}})
    check(f"{label}: decide_choice", not r.is_error and d["choice"] in ("billing", "tech", "sales") and abs(sum(d["probabilities"].values()) - 1) < 1e-3,
          f"choice={d.get('choice')} p={d.get('probability', 0):.3f}")
    r, d = await call("decide_score", {"question": "How urgent is this?", "context": "Production database is down, all customers affected.",
                                      "levels": ["low: can wait", "medium: this week", "high: today", "critical: now"]})
    check(f"{label}: decide_score", not r.is_error and 0 <= d["score"] <= 3 and d["level"].split(":")[0] in ("low", "medium", "high", "critical"),
          f"score={d.get('score')} level={d.get('level')!r}")
    r, d = await call("decide_many", {"context": CONTEXT, "questions": {
        "refund": {"type": "noul", "instructions": "Can the customer get a refund?"},
        "channel": {"type": "choice", "instructions": "Best channel to reply?", "criteria": {"email": "", "phone": ""}}}})
    check(f"{label}: decide_many", not r.is_error and set(d["answers"]) == {"refund", "channel"}, str(list(d.get("answers", {}))))
    r, d = await call("engine_info", {})
    check(f"{label}: engine_info", not r.is_error and "prompt_hash" in d, f"model={d.get('model')}")
    r, d = await call("decide_choice", {"question": "x", "options": ["only-one"]})
    check(f"{label}: bad input is a tool error", r.is_error, d.get("error", "")[:60])


async def http_run(token):
    from mcp import ClientSession
    from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client
    async with create_mcp_http_client(headers={"Authorization": f"Bearer {token}"}) as http:
        async with streamable_http_client(f"http://127.0.0.1:{PORT}/mcp", http_client=http) as streams:
            read, write = streams[0], streams[1]
            async with ClientSession(read, write) as s:
                await s.initialize()
                await exercise(s, "http")


async def stdio_run():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    params = StdioServerParameters(command=sys.executable, args=[SERVER, "--transport", "stdio"],
                                   env={**os.environ, "RYOTIDE_URL": API})
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as s:
            await s.initialize()
            await exercise(s, "stdio")


def main():
    token = secrets.token_urlsafe(24)
    env = {**os.environ, "RYOTIDE_URL": API, "RYOTIDE_MCP_TOKEN": token}
    proc = subprocess.Popen([sys.executable, SERVER, "--port", str(PORT)], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    try:
        url = f"http://127.0.0.1:{PORT}/mcp"
        for _ in range(60):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/", timeout=1)
            except urllib.error.HTTPError:
                break
            except Exception:
                time.sleep(0.5)
        check("http: 401 without a token", status_of(url, {}) == 401)
        check("http: 401 with a wrong token", status_of(url, {"Authorization": "Bearer wrong"}) == 401)
        asyncio.run(http_run(token))
    finally:
        proc.terminate(); proc.wait(timeout=10)
    # forward-auth with the decision API unreachable: refuse with 503, never let the request through
    proc = subprocess.Popen([sys.executable, SERVER, "--port", str(PORT), "--forward-auth"],
                            env={**os.environ, "RYOTIDE_URL": "http://127.0.0.1:9"}, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/", timeout=1)
            except urllib.error.HTTPError:
                break
            except Exception:
                time.sleep(0.5)
        check("http: forward-auth, API down -> 503", status_of(f"http://127.0.0.1:{PORT}/mcp", {"Authorization": "Bearer x"}) == 503)
    finally:
        proc.terminate(); proc.wait(timeout=10)
    asyncio.run(stdio_run())
    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
