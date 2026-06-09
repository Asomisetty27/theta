"""
Theta MCP server — exposes live daemon state as Claude tools.

Implements the Model Context Protocol (stdio transport, JSON-RPC 2.0).
Reads from the Theta health API (port 9102 by default). Falls back to
a "daemon unreachable" response rather than erroring, so the tool still
works when Theta isn't running (returns last-known or error state).

Add to .mcp.json:
  {
    "mcpServers": {
      "theta": {
        "command": "python",
        "args": ["/Users/amogh/thermalos-agent/theta/mcp_server.py"],
        "env": {"THETA_PORT": "9102"}
      }
    }
  }

Tools exposed:
  theta_fleet_status   — R_θ, risk, state for all GPUs
  theta_gpu_details    — causal explanation, fault, maintenance for one GPU
  theta_gpu_risk       — quick risk score for a GPU (sub-second)
  theta_fleet_summary  — one-line narrative: "3 GPUs nominal, GPU 2 drifting"
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

THETA_PORT = int(os.environ.get("THETA_PORT", "9102"))
SERVER_NAME = "theta-mcp"
SERVER_VERSION = "0.1.0"

TOOLS = [
    {
        "name": "theta_fleet_status",
        "description": (
            "Get current thermal state for all GPUs from the running Theta daemon. "
            "Returns R_θ (thermal resistance C/W), risk score, state (idle/load/drifting/critical), "
            "and fault classification for each GPU."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "theta_gpu_details",
        "description": (
            "Get rich diagnostic detail for a single GPU: causal explanation of why it's in its "
            "current state, fault curve analysis, maintenance score, and CNN prediction. "
            "Use this when a GPU shows anomalous risk or state."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "gpu_index": {
                    "type": "integer",
                    "description": "Zero-based GPU index (0 = first GPU)",
                }
            },
            "required": ["gpu_index"],
        },
    },
    {
        "name": "theta_gpu_risk",
        "description": (
            "Get the risk score (0.0–1.0) for a specific GPU. "
            "0.0 = nominal, >0.5 = watch, >0.8 = act immediately."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "gpu_index": {
                    "type": "integer",
                    "description": "Zero-based GPU index",
                }
            },
            "required": ["gpu_index"],
        },
    },
    {
        "name": "theta_fleet_summary",
        "description": (
            "Get a one-sentence narrative summary of fleet health. "
            "E.g. '7 GPUs nominal, GPU 3 drifting (R_θ=0.074, risk=0.81, fault=cooling_degradation)'. "
            "Use this for quick status checks before diving into details."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]


def _api(path: str) -> dict:
    url = f"http://localhost:{THETA_PORT}{path}"
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            return json.loads(r.read().decode())
    except urllib.error.URLError as e:
        return {"error": str(e), "daemon_status": "unreachable", "port": THETA_PORT}
    except Exception as e:
        return {"error": str(e)}


def _fleet_summary(status: dict) -> str:
    if "error" in status:
        return f"Theta daemon unreachable on port {THETA_PORT}. Start with: theta monitor"
    gpus = status.get("gpus", [])
    if not gpus:
        return "No GPUs found."
    nominal = [g for g in gpus if g.get("state") in ("clean_idle", "under_load", "idle", "load")]
    anomalous = [g for g in gpus if g.get("state") not in ("clean_idle", "under_load", "idle", "load", None)]
    parts = []
    if nominal:
        parts.append(f"{len(nominal)} GPU{'s' if len(nominal) > 1 else ''} nominal")
    for g in anomalous:
        idx = g.get("index", g.get("gpu_index", "?"))
        state = g.get("state", "unknown")
        rtheta = g.get("rtheta_cw", g.get("rtheta", None))
        risk = g.get("risk_score", None)
        fault = g.get("fault_class", None)
        desc = f"GPU {idx} {state}"
        if rtheta is not None:
            desc += f" (R_θ={rtheta:.3f}"
        if risk is not None:
            desc += f", risk={risk:.2f}"
        if fault:
            desc += f", fault={fault}"
        if rtheta is not None:
            desc += ")"
        parts.append(desc)
    return ", ".join(parts) if parts else "Fleet status unknown."


def _handle(request: dict) -> dict:
    method = request.get("method", "")

    if method == "initialize":
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        }

    if method == "notifications/initialized":
        return {}

    if method == "tools/list":
        return {"tools": TOOLS}

    if method == "tools/call":
        name = request.get("params", {}).get("name", "")
        args = request.get("params", {}).get("arguments", {})

        if name == "theta_fleet_status":
            result = _api("/api/v1/agent/fleet/status")

        elif name == "theta_gpu_details":
            idx = args.get("gpu_index", 0)
            result = _api(f"/api/v1/agent/gpu/{idx}/details")

        elif name == "theta_gpu_risk":
            idx = args.get("gpu_index", 0)
            health = _api(f"/api/v1/health/gpu/{idx}")
            if "error" in health:
                result = health
            else:
                result = {
                    "gpu_index": idx,
                    "risk_score": health.get("risk_score"),
                    "state": health.get("state"),
                    "rtheta_cw": health.get("rtheta_cw"),
                }

        elif name == "theta_fleet_summary":
            status = _api("/api/v1/agent/fleet/status")
            result = {"summary": _fleet_summary(status)}

        else:
            result = {"error": f"Unknown tool: {name}"}

        return {
            "content": [{"type": "text", "text": json.dumps(result, indent=2)}]
        }

    # Unhandled method — return empty result (not an error)
    return {}


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue

        result = _handle(req)

        # notifications don't get a response
        if req.get("method", "").startswith("notifications/"):
            continue

        resp = {
            "jsonrpc": "2.0",
            "id": req.get("id"),
            "result": result,
        }
        print(json.dumps(resp), flush=True)


if __name__ == "__main__":
    main()
