#!/usr/bin/env python3
"""
NM i AI 2026 — Tripletex AI Accounting Agent
Flask endpoint: POST /solve
Set ANTHROPIC_KEY environment variable before running.
"""
import os, json, base64, requests
from collections import deque
from flask import Flask, request, jsonify
from anthropic import Anthropic

app = Flask(__name__)
client = Anthropic(api_key=os.environ.get("ANTHROPIC_KEY", ""))

# In-memory log of last 3 executions
EXEC_LOGS = deque(maxlen=3)

SYSTEM_PROMPT = """You are an expert Tripletex accounting agent. You receive a task in any language (Norwegian, English, Spanish, Portuguese, Nynorsk, German, French) and must complete it using the Tripletex v2 REST API.

IMPORTANT RULES:
- Always use the provided base_url and session_token — never hardcode URLs
- Authenticate with Basic Auth: username="0", password=session_token
- You MUST use tools for every action — never respond with just text
- Think step by step, make the required API calls, then call finish_task when ALL steps are done
- List resources first if you need to find IDs
- If an API call fails with an error, analyze the error and try again with corrected parameters
- If you get a 401/403, the auth is invalid — try the same call once more then move on
- If you get a 404, the resource path may be wrong — try variations
- If you get a 400, check the required fields and try again with corrected body

COMMON PATTERNS:
- Create employee: POST /employee {firstName, lastName, email, roleAsAccountant:true}
- Create customer: POST /customer {name, email, isCustomer:true}
- Create supplier: POST /customer {name, isSupplier:true}
- Create product: POST /product {name, number, costExcludingVatCurrency}
- Create department: POST /department {name}
- Create project: POST /project {name, startDate, customer:{id:X}}
- Create order: POST /order {customer:{id:X}, orderDate, orderLines:[{product:{id:X}, count, unitPriceExcludingVat}]}
- Create invoice from order: POST /invoice {invoiceDate, customer:{id:X}, orders:[{id:X}]}
- Invoice payment types: GET /invoice/paymentType
- Register payment: PUT /invoice/{id}/:pay?paymentDate=YYYY-MM-DD&paymentTypeId=X&amount=Y
- Credit note: POST /invoice/{id}/:createCreditNote?date=YYYY-MM-DD
- Travel expense: POST /travelExpense {employee:{id:X}, startDate, endDate, destination, description}
- Delete resource: DELETE /endpoint/{id}
- List: GET /endpoint?fields=id,name,*&count=100

RESPONSE FORMAT:
- Lists: {"values": [...], "fullResultSize": N}
- Single: {"value": {...}}
- Use ?fields=* for all fields
- Dates: YYYY-MM-DD

TROUBLESHOOTING:
- If POST /employee returns 400: try adding roleAsAccountant:true to the body
- If POST /customer returns 400: ensure isCustomer:true or isSupplier:true is set
- If POST /order returns 400: orderLines must be a non-empty array with product.id, count, and unitPriceExcludingVat
- If PUT /:pay returns 400: check paymentTypeId is valid via GET /invoice/paymentType first

When ALL steps of the task are complete, call finish_task with a summary."""

TOOLS = [
    {
        "name": "api_get",
        "description": "GET request to Tripletex API",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "e.g. /employee?fields=id,firstName,lastName"},
            },
            "required": ["path"]
        }
    },
    {
        "name": "api_post",
        "description": "POST request to create a resource",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "body": {"type": "object"}
            },
            "required": ["path", "body"]
        }
    },
    {
        "name": "api_put",
        "description": "PUT request to update a resource or trigger an action (e.g. /invoice/123/:pay?paymentDate=2025-01-01&paymentTypeId=1&amount=100)",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "body": {"type": "object", "description": "JSON body — use {} for action endpoints with query params"}
            },
            "required": ["path", "body"]
        }
    },
    {
        "name": "api_delete",
        "description": "DELETE request to remove a resource",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "e.g. /employee/123"}
            },
            "required": ["path"]
        }
    },
    {
        "name": "finish_task",
        "description": "Call ONLY when ALL task steps are fully completed",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"}
            },
            "required": ["summary"]
        }
    }
]


def call_api(method, base_url, session_token, path, body=None):
    base = base_url.rstrip("/")
    url  = base + (path if path.startswith("/") else "/" + path)
    auth = ("0", session_token)
    hdrs = {"Content-Type": "application/json", "Accept": "application/json"}
    print(f"[api] {method} {url}")
    try:
        if method == "GET":
            r = requests.get(url, auth=auth, headers=hdrs, timeout=30)
        elif method == "POST":
            r = requests.post(url, auth=auth, headers=hdrs, json=body or {}, timeout=30)
        elif method == "PUT":
            r = requests.put(url, auth=auth, headers=hdrs, json=body or {}, timeout=30)
        elif method == "DELETE":
            r = requests.delete(url, auth=auth, headers=hdrs, timeout=30)
        print(f"[api] status={r.status_code}")
        if r.status_code == 204:
            return {"status": "deleted"}
        try:
            return r.json()
        except:
            return {"status": r.status_code, "text": r.text[:500]}
    except Exception as e:
        return {"error": str(e)}


def run_agent(prompt, files, base_url, session_token):
    exec_log = {"prompt": prompt[:300], "base_url": base_url, "calls": [], "iterations": 0}
    EXEC_LOGS.append(exec_log)
    content = [{"type": "text", "text": f"TASK: {prompt}"}]
    for f in files:
        if f.get("mime_type", "").startswith("image/"):
            content.append({"type": "image", "source": {"type": "base64", "media_type": f["mime_type"], "data": f["content_base64"]}})
        else:
            try:
                text = base64.b64decode(f["content_base64"]).decode("utf-8", errors="replace")[:3000]
                content.append({"type": "text", "text": f"\nFile [{f['filename']}]:\n{text}"})
            except:
                pass

    messages = [{"role": "user", "content": content}]

    for iteration in range(30):
        exec_log["iterations"] = iteration + 1
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=8096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            tool_choice={"type": "any"},
            messages=messages
        )
        print(f"[agent] iter={iteration} stop_reason={response.stop_reason}")
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        done = False
        for block in response.content:
            if block.type != "tool_use":
                continue
            inp = block.input
            print(f"[agent] tool={block.name} inp={json.dumps(inp)[:200]}")
            if block.name == "finish_task":
                done = True
                res = {"done": True, "summary": inp.get("summary", "")}
                exec_log["calls"].append({"tool": "finish_task", "summary": inp.get("summary", "")[:200]})
                print(f"[agent] FINISHED: {inp.get('summary', '')[:200]}")
            elif block.name == "api_get":
                res = call_api("GET", base_url, session_token, inp["path"])
                exec_log["calls"].append({"tool": "GET", "path": inp["path"], "status": res.get("status", str(res)[:100])})
            elif block.name == "api_post":
                res = call_api("POST", base_url, session_token, inp["path"], inp.get("body", {}))
                exec_log["calls"].append({"tool": "POST", "path": inp["path"], "body_keys": list(inp.get("body", {}).keys()), "result": str(res)[:200]})
            elif block.name == "api_put":
                res = call_api("PUT", base_url, session_token, inp["path"], inp.get("body", {}))
                exec_log["calls"].append({"tool": "PUT", "path": inp["path"], "result": str(res)[:200]})
            elif block.name == "api_delete":
                res = call_api("DELETE", base_url, session_token, inp["path"])
                exec_log["calls"].append({"tool": "DELETE", "path": inp["path"], "result": str(res)[:200]})
            else:
                res = {"error": "unknown tool"}

            tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(res, ensure_ascii=False)[:3000]})

        if done:
            break
        messages.append({"role": "user", "content": tool_results})

    return {"status": "completed"}


@app.route("/solve", methods=["POST"])
def solve():
    data = request.json or {}
    creds = data.get("tripletex_credentials", {})
    base_url = creds.get("base_url", "")
    session_token = creds.get("session_token", "")
    prompt = data.get("prompt", "")
    print(f"[solve] prompt: {prompt[:150]}")
    print(f"[solve] base_url: {base_url}")
    print(f"[solve] token present: {bool(session_token)}")
    print(f"[solve] ANTHROPIC_KEY set: {bool(os.environ.get('ANTHROPIC_KEY'))}")
    try:
        result = run_agent(prompt, data.get("files", []), base_url, session_token)
        print(f"[solve] done: {result}")
        return jsonify(result)
    except Exception as e:
        import traceback
        print(f"[error] {traceback.format_exc()}")
        return jsonify({"status": "completed"})


@app.route("/logs", methods=["GET"])
def logs():
    return jsonify(list(EXEC_LOGS))


@app.route("/health", methods=["GET"])
def health():
    key_ok = bool(os.environ.get("ANTHROPIC_KEY"))
    return jsonify({"ok": True, "anthropic_key_set": key_ok})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)))
