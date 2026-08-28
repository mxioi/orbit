"""Turnstone REST client -- plain HTTP + polling, no SSE (see ask() docstring
for why). Shared by voice_test.py (push-to-talk CLI) and assistant_app.py
(VAD-driven GUI).
"""
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

CONSOLE_BASE = "http://REDACTED-LAN-IP:8095/v1/api"

# Never hardcode the token here -- it goes into git now. Set it via env var,
# or drop it in a local .turnstone_token file (gitignored) as a fallback.
TURNSTONE_TOKEN = os.environ.get("TURNSTONE_TOKEN", "")
if not TURNSTONE_TOKEN:
    _token_file = os.path.join(os.path.dirname(__file__), ".turnstone_token")
    if os.path.isfile(_token_file):
        TURNSTONE_TOKEN = open(_token_file, encoding="utf-8").read().strip()
if not TURNSTONE_TOKEN:
    raise SystemExit(
        "No Turnstone token found. Set the TURNSTONE_TOKEN env var, or put it "
        "(just the raw ts_... value, nothing else) in a .turnstone_token file "
        "next to this script."
    )

TURNSTONE_MODEL = "voice-fast"  # Qwen3.5-9B, reasoning disabled -- confirmed ~1.6s/turn once warm
TURNSTONE_PERSONA = "researcher"  # matches "quizzing me / tech info" use case, not orchestration
HEADERS = {"Authorization": f"Bearer {TURNSTONE_TOKEN}", "Content-Type": "application/json"}

# Set after create_conversation() -- interactive workstreams live on a
# node (turnstone-server), not in-process on the console like
# coordinators. Events/send must go to the node directly.
TURNSTONE_BASE = CONSOLE_BASE


def _call(method, path, body=None, params=None, base=None):
    url = (base or TURNSTONE_BASE) + path
    if params:
        url += "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode()
        return resp.status, (json.loads(raw) if raw else {})


def create_conversation(model="", persona=TURNSTONE_PERSONA, first_message=""):
    """Create a real interactive workstream (not a coordinator) routed to
    an actual turnstone-server node. Sets the module-level TURNSTONE_BASE
    to that node's own URL -- events/send for interactive workstreams
    must go directly to the owning node, not the console's proxy.

    Unlike the coordinator path, initial_message here did NOT reliably
    auto-send in testing (message_count stayed 0) -- so this always
    explicitly sends via watch-then-send on the node, same ordering as
    the coordinator fix, which is not vulnerable to the coordinator's
    worker-thread startup race since interactive workstreams don't queue
    at a "tool-result seam."
    """
    global TURNSTONE_BASE
    # auto_approve_tools (not blanket auto_approve) -- persists for every
    # future tool call on this ws, but scoped to exactly the researcher
    # persona's own tool_allowlist (read-only/informational: no bash, no
    # writes, nothing destructive). A blanket auto_approve=True would also
    # silently cover any tool a future persona/config change adds, which
    # is a broader grant than this task actually needs.
    body = {
        "persona": persona,
        "auto_approve_tools": [
            "read_file", "search", "web_fetch", "web_search", "recall",
            "memory", "tool_search",
        ],
    }
    if model:
        body["model"] = model
    status, resp = _call("POST", "/route/workstreams/new", body, base=CONSOLE_BASE)
    if status not in (200, 201):
        raise SystemExit(f"create failed: {status} {resp}")
    ws_id = resp["ws_id"]
    node_url = resp["node_url"]

    # The console picks node_url and hands it back -- don't blindly trust
    # a server-supplied URL as the target for every subsequent authenticated
    # call (that's a straight SSRF/token-exfiltration path if the console
    # were ever compromised or pointed at something untrusted). Constrain
    # to hosts we actually expect this homelab's single node to report.
    node_host = urllib.parse.urlparse(node_url).hostname
    allowed_hosts = {"turnstone", "REDACTED-LAN-IP", "127.0.0.1", "localhost"}
    if node_host not in allowed_hosts:
        raise SystemExit(
            f"refusing to send credentials to unexpected node_url host {node_host!r} "
            f"(expected one of {allowed_hosts}) -- got node_url={node_url!r}"
        )

    TURNSTONE_BASE = node_url.rstrip("/") + "/v1/api"
    if first_message:
        return ws_id, ask(ws_id, first_message)
    return ws_id, ""


def ask(ws_id, message, timeout_s=120, poll_interval=0.5, on_poll=None):
    """Send a message and poll for the response instead of trusting SSE.

    SSE turned out to have a real subscribe/publish race that survived
    multiple fix attempts (confirmed on both coordinator AND plain
    interactive workstreams -- not kind-specific, a race in the event
    stream itself), causing responses to arrive empty or bleed into the
    NEXT turn's window. History polling sidesteps it entirely: /send is
    fire-and-forget, and /cluster/ws/{id}/detail's `live.state` plus
    `tail` reliably reflect ground truth regardless of SSE delivery.
    Trades away live token-by-token progress for actually being correct
    every time -- the right trade for a voice assistant that only needs
    the final text to speak, not a live-updating transcript.

    on_poll: optional zero-arg callback invoked once per poll tick (GUI
    callers use this instead of the CLI's print('.') to keep the caller
    from needing to know about polling internals).
    """
    # limit must cover the WHOLE conversation so far, not just the last
    # message -- with a small limit, older assistant rows fall outside
    # ids_before and look "new" to the completion check below, so the
    # poll loop can grab a stale reply from 2+ turns back before the
    # actual fresh one exists yet. Confirmed live: turns 3 and 4 of a
    # 4-turn conversation returned turn 1 and 2's exact text, instantly.
    _, hist_before = _call("GET", f"/workstreams/{ws_id}/history", params={"limit": 1000})
    ids_before = {m.get("event_id") for m in hist_before.get("messages", [])}

    _call("POST", f"/workstreams/{ws_id}/send", {"message": message})

    t0 = time.time()
    while time.time() - t0 < timeout_s:
        time.sleep(poll_interval)
        if on_poll:
            on_poll()
        _, detail = _call(
            "GET", f"/cluster/ws/{ws_id}/detail", params={"message_limit": 15}, base=CONSOLE_BASE
        )
        # A new assistant row with NON-EMPTY content is the real completion
        # signal. The researcher persona auto-checks a "memory pointer" on
        # nearly every turn -- that shows up as an assistant row that's
        # PURELY a tool call (content="", real payload in tool_calls),
        # followed later by the actual text-bearing assistant row after the
        # tool result comes back. Confirmed live: grabbing the tool-call
        # row as "the answer" returns '' even though the real answer
        # completes moments later in the same turn -- skip empty rows and
        # keep polling for the one that actually has text.
        # NOTE: this endpoint's message dicts use "messages" (not "tail"
        # per the docs) and "_event_id" (underscore-prefixed, unlike
        # /history's bare "event_id") -- confirmed against a live response.
        for msg in reversed(detail.get("messages", [])):
            if (msg.get("role") == "assistant" and msg.get("_event_id") not in ids_before
                    and msg.get("content", "").strip()):
                return msg.get("content", "")

    return ""


def close_conversation(ws_id):
    try:
        _call("POST", f"/workstreams/{ws_id}/close", {})
    except urllib.error.URLError:
        pass
