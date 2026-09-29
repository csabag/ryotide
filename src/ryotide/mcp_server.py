# /// script
# requires-python = ">=3.12"
# dependencies = ["mcp>=2.2,<3", "uvicorn>=0.30"]
# ///
"""RYOTIDE as an MCP server: typed decisions (choice, yes/no, score) as tools.

A thin client of a running RYOTIDE decision API (POST /v1/systemone); it loads no model and
needs only the MCP SDK, so it runs anywhere with `uv run src/ryotide/mcp_server.py`.

Transports:
  http   (default) Streamable HTTP at /mcp -- the remote transport of the MCP spec. Stateless,
         plain JSON responses (no long-lived streams: decisions take well under a second, and
         proxies such as RunPod's cut connections at ~100 s).
  stdio  for a client that launches the server as a local process.

Auth (HTTP; required unless bound to loopback):
  RYOTIDE_MCP_TOKEN=<secret>   clients send 'Authorization: Bearer <secret>'; tool calls use
                               RYOTIDE_API_KEY (if any) towards the decision API.
  --forward-auth               clients send their own decision-API key; it is checked against the
                               API on every request (GET /v1/auth when the API has it: no rate-limit
                               token spent; else /health) and forwarded on every call, so the API's
                               keys, revocation and rate limits apply at once. The API must enforce keys.
  An unreachable or failing API answers 503, never lets a request through.

  RYOTIDE_URL      decision API base URL (default http://127.0.0.1:8778)
  RYOTIDE_API_KEY  bearer key for the API (stdio and token modes)

    uv run src/ryotide/mcp_server.py --port 8889                                   # loopback, no auth
    RYOTIDE_MCP_TOKEN=... uv run src/ryotide/mcp_server.py --host 0.0.0.0 --port 8889
    uv run src/ryotide/mcp_server.py --transport stdio
"""
from __future__ import annotations

import argparse
import hmac
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

API_URL = os.environ.get("RYOTIDE_URL", "http://127.0.0.1:8778").rstrip("/")
API_KEY = os.environ.get("RYOTIDE_API_KEY", "").strip()
MCP_TOKEN = os.environ.get("RYOTIDE_MCP_TOKEN", "").strip()
TIMEOUT = float(os.environ.get("RYOTIDE_TIMEOUT", "90"))
FORWARD_AUTH = False                     # set by --forward-auth

INSTRUCTIONS = """RYOTIDE answers typed decisions with calibrated probabilities from one forward pass of a
local LLM (no text generation). Put everything the decision depends on into `context` (a document, a
policy, a record as text or JSON) and ask one precise question. Use decide_choice for picking one of
several options (give each option a short description), decide_yes_no for a binary judgement, and
decide_score for an ordinal rating on levels you define. The probabilities are calibrated: pass
review_threshold (e.g. 0.8) to get needs_review=true on low-confidence answers and route those to a
human. It is weak at multi-step arithmetic and date arithmetic; compute those first and put the
result into the context."""

server = MCPServer("ryotide", instructions=INSTRUCTIONS)


# ---------------------------------------------------------------------------------------- API client

def _call(method: str, path: str, body: dict | None = None, auth: str | None = None) -> tuple[int, Any]:
    req = urllib.request.Request(API_URL + path, method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json", "Accept": "application/json"})
    if auth:
        req.add_header("Authorization", auth)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"null")
        except ValueError:
            return e.code, None
    except (urllib.error.URLError, TimeoutError) as e:
        raise ToolError(f"decision API unreachable at {API_URL}: {getattr(e, 'reason', e)}") from None


def _auth_for(ctx: Context | None) -> str | None:
    """The Authorization header to send to the decision API."""
    if FORWARD_AUTH:
        h = (ctx.headers or {}) if ctx is not None else {}
        return h.get("authorization") or h.get("Authorization")
    return f"Bearer {API_KEY}" if API_KEY else None


def _decide(state: Any, questions: dict, review_threshold: float | None, ctx: Context | None) -> dict[str, Any]:
    body: dict[str, Any] = {"model": "ryotide", "state": state if state is not None else "", "questions": questions}
    if review_threshold is not None:
        if not 0 <= review_threshold <= 1:
            raise ToolError("review_threshold must be between 0 and 1")
        body["review_threshold"] = review_threshold
    status, out = _call("POST", "/v1/systemone", body, _auth_for(ctx))
    if status == 401:
        raise ToolError("the decision API refused the key (401)")
    if status == 429:
        raise ToolError("rate limit reached for this key (429); retry shortly")
    if status != 200 or not isinstance(out, dict):
        msg = out.get("error") if isinstance(out, dict) else None
        raise ToolError(f"decision API error {status}: {msg or 'no details'}")
    return out


def _review_fields(ans: dict, confidence: float, review_threshold: float | None) -> dict[str, Any]:
    """confidence = probability of the chosen answer (yes/no: the likelier side). Servers with the
    review feature return these fields themselves; otherwise they are computed here the same way."""
    if review_threshold is None:
        return {}
    conf = float(ans.get("confidence", confidence))
    return {"confidence": conf, "needs_review": bool(ans.get("needs_review", conf < review_threshold))}


def _latency(out: dict) -> dict[str, Any]:
    rt = out.get("runtime") or {}
    return {"latency_ms": round(rt["latency_s"] * 1000, 1)} if "latency_s" in rt else {}


# ---------------------------------------------------------------------------------------- tools

READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False)


@server.tool(annotations=READ_ONLY, structured_output=True)
def decide_choice(question: str, options: dict[str, str] | list[str], context: str | dict | None = None,
                  review_threshold: float | None = None, ctx: Context | None = None) -> dict[str, Any]:
    """Pick exactly one of several options, with a calibrated probability for every option.

    question: what to decide, e.g. "Which team should handle this ticket?"
    options: {label: description} (recommended; labels are short ids such as "billing") or a list
      of labels. 2 to 260 options.
    context: the material the decision is based on (text, or a JSON object).
    review_threshold: optional 0-1; answers below it come back with needs_review=true.
    """
    criteria = options if isinstance(options, dict) else {str(o): "" for o in options}
    if len(criteria) < 2:
        raise ToolError("options needs at least two entries")
    out = _decide(context, {"q": {"type": "choice", "instructions": question, "criteria": criteria}}, review_threshold, ctx)
    ans = out["answers"]["q"]
    probs = dict(sorted(ans["probabilities"].items(), key=lambda kv: -kv[1]))
    return {"choice": ans["choice"], "probability": probs[ans["choice"]], "probabilities": probs,
            **_review_fields(ans, probs[ans["choice"]], review_threshold), **_latency(out)}


@server.tool(annotations=READ_ONLY, structured_output=True)
def decide_yes_no(question: str, context: str | dict | None = None, review_threshold: float | None = None,
                  ctx: Context | None = None) -> dict[str, Any]:
    """Answer a yes/no question with the calibrated probability of "yes".

    question: a question that has a yes or no answer, e.g. "Does this claim fall under the policy?"
    context: the material the answer is based on (text, or a JSON object).
    review_threshold: optional 0-1; answers whose likelier side is below it get needs_review=true.
    """
    out = _decide(context, {"q": {"type": "noul", "instructions": question}}, review_threshold, ctx)
    ans = out["answers"]["q"]
    p = float(ans["noul"])
    return {"answer": "yes" if p >= 0.5 else "no", "p_yes": p, **_review_fields(ans, max(p, 1 - p), review_threshold),
            **_latency(out)}


@server.tool(annotations=READ_ONLY, structured_output=True)
def decide_score(question: str, levels: list[str], context: str | dict | None = None,
                 review_threshold: float | None = None, ctx: Context | None = None) -> dict[str, Any]:
    """Rate on an ordinal scale you define, with a probability for every level.

    question: what to rate, e.g. "How urgent is this request?"
    levels: the scale from lowest to highest, each described, e.g.
      ["not urgent: can wait a week", "normal: this week", "urgent: today", "critical: now"]. 2 to 10.
    context: the material the rating is based on.
    review_threshold: optional 0-1; answers below it come back with needs_review=true.
    Returns score (0-based index into levels), the level text, and the probabilities.
    """
    if not 2 <= len(levels) <= 10:
        raise ToolError("levels needs 2 to 10 entries")
    out = _decide(context, {"q": {"type": "score", "instructions": question, "criteria": list(levels)}},
                  review_threshold, ctx)
    ans = out["answers"]["q"]
    i = int(ans["score"])
    probs = {levels[int(k)]: v for k, v in ans["probabilities"].items()}
    expected = sum(int(k) * v for k, v in ans["probabilities"].items())
    return {"score": i, "level": levels[i], "expected_score": round(expected, 3), "probabilities": probs,
            **_review_fields(ans, float(ans["probabilities"][str(i)]), review_threshold), **_latency(out)}


@server.tool(annotations=READ_ONLY, structured_output=True)
def decide_many(context: str | dict, questions: dict[str, dict], review_threshold: float | None = None,
                ctx: Context | None = None) -> dict[str, Any]:
    """Several typed questions about the same context in one call (the context is read once).

    questions: {key: question} in the TypeSafe /v1/systemone format, e.g.
      {"eligible": {"type": "noul", "instructions": "Is the customer eligible?"},
       "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "...", "tech": "..."}},
       "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "medium", "high"]}}
    Returns the API's answers per key (choice + probabilities, noul = P(yes), score + probabilities).
    """
    out = _decide(context, questions, review_threshold, ctx)
    return {"answers": out.get("answers", {}), **_latency(out)}


@server.tool(annotations=READ_ONLY, structured_output=True)
def engine_info(ctx: Context | None = None) -> dict[str, Any]:
    """The model, revision, decision configuration and prompt hash behind these tools."""
    status, out = _call("GET", "/health", auth=_auth_for(ctx))
    if status != 200 or not isinstance(out, dict):
        raise ToolError(f"decision API /health returned {status}")
    return out


# ---------------------------------------------------------------------------------------- HTTP auth

class BearerGate:
    """ASGI middleware: every HTTP request needs a valid 'Authorization: Bearer ...'. No caching: a
    revoked key is refused on its next request. If the key cannot be checked (API down), 503."""

    MESSAGES = {401: "missing or invalid bearer token", 429: "rate limited",
                503: "decision API unavailable; cannot check the key -- retry shortly"}

    def __init__(self, app, check):
        self.app, self.check = app, check

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        header = dict(scope.get("headers") or []).get(b"authorization", b"").decode("latin-1")
        status = self._verdict(header)
        if status == 200:
            return await self.app(scope, receive, send)
        hdrs = [(b"content-type", b"application/json")]
        if status == 401:
            hdrs.append((b"www-authenticate", b'Bearer realm="ryotide"'))
        elif status in (429, 503):
            hdrs.append((b"retry-after", b"5"))
        await send({"type": "http.response.start", "status": status, "headers": hdrs})
        await send({"type": "http.response.body", "body": json.dumps({"error": self.MESSAGES[status]}).encode()})

    def _verdict(self, header: str) -> int:
        if not header.startswith("Bearer ") or not header[7:].strip():
            return 401
        try:
            return self.check(header)
        except Exception:                # API unreachable or broken: fail closed, but say why
            return 503


def _static_check(header: str) -> int:
    return 200 if hmac.compare_digest(header[7:].strip().encode(), MCP_TOKEN.encode()) else 401


def _forward_check(header: str) -> int:
    status, _ = _call("GET", "/v1/auth", auth=header)
    if status == 404:                    # an API without /v1/auth: /health (spends a rate-limit token)
        status, _ = _call("GET", "/health", auth=header)
    if status in (200, 429):
        return status
    return 503 if status >= 500 else 401


def _api_enforces_keys() -> bool:
    try:
        status, _ = _call("GET", "/health")
    except ToolError:
        return True                      # unreachable now; the per-request check decides later
    return status == 401


def main() -> None:
    global FORWARD_AUTH
    ap = argparse.ArgumentParser(description="RYOTIDE decisions as an MCP server")
    ap.add_argument("--transport", choices=("http", "stdio"), default="http")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8889)
    ap.add_argument("--path", default="/mcp")
    ap.add_argument("--forward-auth", action="store_true",
                    help="clients authenticate with their own decision-API key, which is forwarded")
    a = ap.parse_args()
    FORWARD_AUTH = a.forward_auth

    if a.transport == "stdio":
        if FORWARD_AUTH:
            raise SystemExit("--forward-auth needs the http transport")
        server.run("stdio")
        return

    import uvicorn
    from mcp.server.transport_security import TransportSecuritySettings

    loopback = a.host in ("127.0.0.1", "localhost", "::1")
    if FORWARD_AUTH:
        if MCP_TOKEN:
            raise SystemExit("use either RYOTIDE_MCP_TOKEN or --forward-auth, not both")
        if not loopback and not _api_enforces_keys():
            raise SystemExit(f"--forward-auth: {API_URL} answers without a key, so forwarding would not "
                             "authenticate anyone; refusing to listen beyond loopback")
        check = _forward_check
    elif MCP_TOKEN:
        check = _static_check
    elif not loopback:
        raise SystemExit("listening beyond loopback needs RYOTIDE_MCP_TOKEN or --forward-auth")
    else:
        check = None

    app = server.streamable_http_app(
        streamable_http_path=a.path, stateless_http=True, json_response=True, host=a.host,
        # Behind a reverse proxy (RunPod) the Host header is the proxy's; the bearer gate protects it.
        transport_security=None if loopback else TransportSecuritySettings(enable_dns_rebinding_protection=False))
    if check is not None:
        app = BearerGate(app, check)
    mode = "forward-auth" if FORWARD_AUTH else "token" if MCP_TOKEN else "no auth (loopback)"
    print(f"ryotide-mcp ready  http://{a.host}:{a.port}{a.path}  api={API_URL}  auth={mode}", file=sys.stderr, flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
