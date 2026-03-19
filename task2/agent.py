#!/usr/bin/env python3
"""
NM i AI 2026 — Tripletex AI Accounting Agent
Flask endpoint: POST /solve
Set ANTHROPIC_KEY environment variable before running.
"""
import os, json, base64, requests
from flask import Flask, request, jsonify
from anthropic import Anthropic

app = Flask(__name__)
client = Anthropic(api_key=os.environ.get("ANTHROPIC_KEY", ""))

SYSTEM_PROMPT = """You are an expert Tripletex accounting agent. You receive a task in any language (Norwegian, English, Spanish, Portuguese, Nynorsk, German, French) and must complete it using the Tripletex v2 REST API.

IMPORTANT RULES:
- Always use the provided base_url and session_token — never hardcode URLs
- Authenticate with Basic Auth: username="0", password=session_token
- Use the tools to make API calls. Think step by step before acting.
- List resources first if you need to find IDs
- Return only after ALL steps of the task are completed

COMMON PATTERNS:
- Create employee: POST /employee {firstName, lastName, email, roleAsAccountant/roleAsAdministrator}
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

When ALL steps are complete, call finish_task."""

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
        "description": "PUT request to update a resource or trigger an action (e.g. /invoice/123/:pay)",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "body": {"type": "object", "description": "JSON body — can be empty {} for action endpoints"}
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
        "description": "Call when ALL task steps are completed",
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
    try:
        if method == "GET":
            r = requests.get(url, auth=auth, headers=hdrs, timeout=30)
        elif method == "POST":
            r = requests.post(url, auth=auth, headers=hdrs, json=body or {}, timeout=30)
        elif method == "PUT":
            r = requests.put(url, auth=auth, headers=hdrs, json=body or {}, timeout=30)
        elif method == "DELETE":
            r = requests.delete(url, auth=auth, headers=hdrs, timeout=30)
        if r.status_code == 204:
            return {"status": "deleted"}
        try:
            return r.json()
        except:
            return {"status": r.status_code, "text": r.text[:500]}
    except Exception as e:
        return {"error": str(e)}


def run_agent(prompt, files, base_url, session_token):
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

    for _ in range(25):
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason in ("end_turn", None) or response.stop_reason != "tool_use":
            break

        tool_results = []
        done = False
        for block in response.content:
            if block.type != "tool_use":
                continue
            inp = block.input
            if block.name == "finish_task":
                done = True
                res = {"done": True, "summary": inp.get("summary", "")}
            elif block.name == "api_get":
                res = call_api("GET", base_url, session_token, inp["path"])
            elif block.name == "api_post":
                res = call_api("POST", base_url, session_token, inp["path"], inp.get("body", {}))
            elif block.name == "api_put":
                res = call_api("PUT", base_url, session_token, inp["path"], inp.get("body", {}))
            elif block.name == "api_delete":
                res = call_api("DELETE", base_url, session_token, inp["path"])
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
    print(f"[solve] {data.get('prompt','')[:100]}")
    try:
        return jsonify(run_agent(data.get("prompt",""), data.get("files",[]), creds.get("base_url",""), creds.get("session_token","")))
    except Exception as e:
        print(f"[error] {e}")
        return jsonify({"status": "completed"})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)))
