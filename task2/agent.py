#!/usr/bin/env python3
"""
NM i AI 2026 — Tripletex AI Accounting Agent
Flask endpoint: POST /solve
Set ANTHROPIC_KEY environment variable before running.
"""
import os, json, base64, requests, time
from flask import Flask, request, jsonify
from anthropic import Anthropic

app = Flask(__name__)
client = Anthropic(api_key=os.environ.get("ANTHROPIC_KEY", ""))

LOG_FILE = "/tmp/agent_logs.json"

def save_log(entry):
    try:
        try:
            logs = json.load(open(LOG_FILE))
        except:
            logs = []
        logs.append(entry)
        logs = logs[-5:]  # keep last 5
        json.dump(logs, open(LOG_FILE, "w"), ensure_ascii=False)
    except Exception as e:
        print(f"[log] save failed: {e}")

SYSTEM_PROMPT = """You are an expert Tripletex accounting agent. You receive a task in any language (Norwegian, English, Spanish, Portuguese, Nynorsk, German, French) and must complete it using the Tripletex v2 REST API.

CRITICAL RULES:
- Always use the provided base_url and session_token — never hardcode URLs
- Authenticate with Basic Auth: username="0", password=session_token
- You MUST use tools for EVERY action — ALWAYS start with an API call, never respond with just text
- Think step by step. Make ALL required API calls. Call finish_task only when EVERY step is done.
- If an API call fails, analyze the error and retry with corrected parameters — do NOT give up

FINDING RESOURCES:
- Find customer by name: GET /customer?name=CustomerName&fields=id,name,organizationNumber
- Find customer by org number: GET /customer?organizationNumber=123456789&fields=id,name
- Find invoices for customer: GET /invoice?customerId=X&fields=id,amount,amountOutstandingCurrency,amountCurrency,invoiceDate,status
- Find employees: GET /employee?fields=id,firstName,lastName,email&count=100
- Find products: GET /product?fields=id,name,number&count=100

CREATING RESOURCES:
- Create employee: POST /employee {"firstName":"X","lastName":"Y","email":"x@y.com","roleAsAccountant":true}
- Create customer: POST /customer {"name":"X","isCustomer":true}
- Create supplier: POST /customer {"name":"X","isSupplier":true}
- Create product: POST /product {"name":"X","number":"001","costExcludingVatCurrency":100}
- Create department: POST /department {"name":"X"}
- Create project: POST /project {"name":"X","startDate":"YYYY-MM-DD","customer":{"id":X}}
- Create order: POST /order {"customer":{"id":X},"orderDate":"YYYY-MM-DD","orderLines":[{"product":{"id":X},"count":1,"unitPriceExcludingVat":100}]}
- Create invoice from order: POST /invoice {"invoiceDate":"YYYY-MM-DD","customer":{"id":X},"orders":[{"id":X}]}

INVOICE PAYMENT (important — follow these steps exactly):
1. Find the invoice: GET /invoice?customerId=X&fields=id,amountOutstandingCurrency,amountCurrency,invoiceDate
2. Get payment types: GET /invoice/paymentType?fields=id,description
3. Choose payment type id (typically the first/default one)
4. Register payment using the invoice's amountOutstandingCurrency (NOT the ex-VAT amount from the task):
   PUT /invoice/{id}/:pay?paymentDate=YYYY-MM-DD&paymentTypeId=X&amount=OUTSTANDING_AMOUNT
   with body: {}

OTHER OPERATIONS:
- Credit note: POST /invoice/{id}/:createCreditNote?date=YYYY-MM-DD with body: {}
- Travel expense: POST /travelExpense {"employee":{"id":X},"startDate":"YYYY-MM-DD","endDate":"YYYY-MM-DD","destination":"X","description":"X"}
- Delete resource: DELETE /endpoint/{id}

RESPONSE FORMAT:
- Lists: {"values": [...], "fullResultSize": N}
- Single: {"value": {...}}
- Always use ?fields=* or specific fields
- Dates: YYYY-MM-DD (use today 2026-03-19 if not specified)

TROUBLESHOOTING:
- POST /employee 400: add "roleAsAccountant":true
- POST /customer 400: ensure "isCustomer":true or "isSupplier":true
- POST /order 400: orderLines must be a non-empty array with product.id, count, unitPriceExcludingVat
- PUT /:pay 400: verify paymentTypeId exists via GET /invoice/paymentType and amount matches outstanding amount
- GET returns empty values: try broader search, omit filters

When ALL steps of the task are fully completed, call finish_task with a detailed summary."""

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
    exec_log = {"ts": time.strftime("%H:%M:%S"), "prompt": prompt[:300], "base_url": base_url, "calls": [], "iterations": 0}
    save_log(exec_log)
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
        response = None
        for attempt in range(4):
            try:
                response = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=8096,
                    system=SYSTEM_PROMPT,
                    tools=TOOLS,
                    messages=messages
                )
                break
            except Exception as api_err:
                err_msg = str(api_err)
                print(f"[agent] Claude API error iter={iteration} attempt={attempt}: {err_msg}")
                exec_log["calls"].append({"tool": "ERROR", "error": err_msg[:300]})
                save_log(exec_log)
                if attempt < 3:
                    time.sleep(3 * (attempt + 1))
        if response is None:
            break
        print(f"[agent] iter={iteration} stop_reason={response.stop_reason}")
        messages.append({"role": "assistant", "content": response.content})

        # If Claude responded with text only (no tools), push it to use tools
        if response.stop_reason == "end_turn":
            has_tool = any(b.type == "tool_use" for b in response.content)
            if not has_tool:
                print(f"[agent] end_turn with no tools — nudging Claude")
                messages.append({"role": "user", "content": "You must use a tool now. Make the first API call required to complete the task."})
                continue

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

        save_log(exec_log)
        if done:
            break
        messages.append({"role": "user", "content": tool_results})

    save_log(exec_log)
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
    try:
        return jsonify(json.load(open(LOG_FILE)))
    except:
        return jsonify([])


@app.route("/health", methods=["GET"])
def health():
    key_ok = bool(os.environ.get("ANTHROPIC_KEY"))
    # Test Anthropic connectivity
    try:
        test_client = Anthropic(api_key=os.environ.get("ANTHROPIC_KEY", ""))
        r = test_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            messages=[{"role": "user", "content": "hi"}]
        )
        claude_ok = True
        claude_err = None
    except Exception as e:
        claude_ok = False
        claude_err = str(e)[:200]
    return jsonify({"ok": True, "anthropic_key_set": key_ok, "claude_reachable": claude_ok, "claude_error": claude_err})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)))
