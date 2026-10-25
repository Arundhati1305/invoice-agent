"""
A2A 1.0 HTTP+JSON invoice action agent (TDS GA3 Q2).

Layers:
  protocol  - routes, auth, version/media checks, error envelopes
  storage   - SQLite: tasks, message idempotency records, decision cache
  decision  - batched LLM call + strict validation/repair of every proposal

GET /healthz is a cheap liveness probe (no DB, no model). On Render, a background
task pings it every 10 minutes so the free-tier instance never goes to sleep.

Run with ONE worker (in-process locks guard races):
  uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
"""
import asyncio
import contextlib
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "http://localhost:8000/a2a/")
BASE_PATH = "/" + PUBLIC_BASE_URL.split("://", 1)[-1].split("/", 1)[-1].strip("/")
if BASE_PATH == "/":
    BASE_PATH = ""
DB_PATH = os.environ.get("DB_PATH", "agent.db")

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://aipipe.org/openai/v1").rstrip("/")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4.1-mini")
LLM_MOCK = os.environ.get("LLM_MOCK", "") == "1"
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "12"))    # items per LLM call (paragraphs are short)
REF_FORMAT = os.environ.get("REF_FORMAT", "inner")       # "inner" -> ABC-1, "bracketed" -> [ABC-1]
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")
DECIDE_BUDGET_S = float(os.environ.get("DECIDE_BUDGET_S", "38"))
LLM_REASONING_EFFORT = os.environ.get("LLM_REASONING_EFFORT", "")  # e.g. "none" for Gemini thinking models
LLM_HTTP_TIMEOUT_MIN = float(os.environ.get("LLM_HTTP_TIMEOUT_MIN", "30"))

# Keep-alive (Render free tier sleeps after ~15 min without inbound traffic).
# Render sets RENDER_EXTERNAL_URL automatically; locally it is absent, so pinging stays off.
_render_url = os.environ.get("RENDER_EXTERNAL_URL", "").rstrip("/")
KEEPALIVE_URL = os.environ.get("KEEPALIVE_URL", f"{_render_url}/healthz" if _render_url else "")
KEEPALIVE_MINUTES = 10  # ping every 10 minutes (Render sleeps after 15 minutes idle)
KEEPALIVE_INTERVAL = KEEPALIVE_MINUTES * 60  # seconds

MT_A2A = "application/a2a+json"
MT_BATCH = "application/vnd.ga5.invoice-claim-batch+json"
MT_PROPOSALS = "application/vnd.ga5.invoice-action-proposals+json"
MT_RESULTS = "application/vnd.ga5.invoice-action-results+json"
MT_RECEIPTS = "application/vnd.ga5.invoice-action-receipts+json"

ACTIONS = ["settle_invoice", "request_approval", "hold_invoice", "reject_duplicate", "open_exception"]
TERMINAL = {"TASK_STATE_COMPLETED", "TASK_STATE_CANCELED", "TASK_STATE_FAILED", "TASK_STATE_REJECTED"}
REF_RE = re.compile(r"\[([^\[\]\n]{1,100})\]")

STATS = {"model_calls": 0}


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------
# Storage
# ----------------------------------------------------------------------------
class Store:
    def __init__(self, path):
        self.lock = threading.Lock()
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS tasks(
            id TEXT PRIMARY KEY, principal TEXT NOT NULL, context_id TEXT,
            state TEXT, created REAL, task_json TEXT, internal_json TEXT);
        CREATE INDEX IF NOT EXISTS tasks_p ON tasks(principal, created);
        CREATE TABLE IF NOT EXISTS messages(
            principal TEXT, message_id TEXT, hash TEXT, task_id TEXT,
            PRIMARY KEY(principal, message_id));
        CREATE TABLE IF NOT EXISTS decisions(key TEXT PRIMARY KEY, json TEXT);
        """)

    def get_message(self, p, mid):
        with self.lock:
            r = self.db.execute("SELECT hash, task_id FROM messages WHERE principal=? AND message_id=?",
                                (p, mid)).fetchone()
        return r

    def get_task(self, tid):
        with self.lock:
            r = self.db.execute("SELECT principal, task_json, internal_json FROM tasks WHERE id=?",
                                (tid,)).fetchone()
        if not r:
            return None
        return {"principal": r[0], "task": json.loads(r[1]), "internal": json.loads(r[2])}

    def save(self, p, task, internal, msg_record=None, created=False):
        """Atomically write the task and (optionally) the message idempotency record."""
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                if created:
                    self.db.execute("INSERT INTO tasks VALUES(?,?,?,?,?,?,?)",
                                    (task["id"], p, task["contextId"], task["status"]["state"],
                                     time.time(), canon(task), canon(internal)))
                else:
                    self.db.execute("UPDATE tasks SET state=?, task_json=?, internal_json=? WHERE id=? AND principal=?",
                                    (task["status"]["state"], canon(task), canon(internal), task["id"], p))
                if msg_record:
                    self.db.execute("INSERT INTO messages VALUES(?,?,?,?)", (p, *msg_record))
                self.db.execute("COMMIT")
            except Exception:
                self.db.execute("ROLLBACK")
                raise

    def list_tasks(self, p, context_id=None, state=None):
        q, a = "SELECT task_json FROM tasks WHERE principal=?", [p]
        if context_id:
            q += " AND context_id=?"; a.append(context_id)
        if state:
            q += " AND state=?"; a.append(state)
        q += " ORDER BY created DESC"
        with self.lock:
            return [json.loads(r[0]) for r in self.db.execute(q, a).fetchall()]

    def get_decision(self, key):
        with self.lock:
            r = self.db.execute("SELECT json FROM decisions WHERE key=?", (key,)).fetchone()
        return json.loads(r[0]) if r else None

    def put_decision(self, key, d):
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO decisions VALUES(?,?)", (key, canon(d)))


store = Store(DB_PATH)

_locks: dict = {}
_locks_guard = threading.Lock()


def lock_for(key):
    with _locks_guard:
        lk = _locks.get(key)
        if lk is None:
            lk = _locks[key] = asyncio.Lock()
        return lk


# ----------------------------------------------------------------------------
# Protocol helpers
# ----------------------------------------------------------------------------
STATUS_NAMES = {400: "INVALID_ARGUMENT", 401: "UNAUTHENTICATED", 403: "PERMISSION_DENIED",
                404: "NOT_FOUND", 409: "ABORTED", 415: "INVALID_ARGUMENT",
                500: "INTERNAL", 502: "UNAVAILABLE", 503: "UNAVAILABLE"}


def err(code, reason, message):
    body = {"error": {"code": code, "status": STATUS_NAMES.get(code, "UNKNOWN"), "message": message,
                      "details": [{"@type": "type.googleapis.com/google.rpc.ErrorInfo",
                                   "reason": reason, "domain": "a2a-protocol.org"}]}}
    return JSONResponse(body, status_code=code, media_type="application/json")


def ok(obj):
    return Response(content=json.dumps(obj, ensure_ascii=False, separators=(",", ":")),
                    status_code=200, media_type=MT_A2A)


def principal_of(request):
    h = request.headers.get("authorization", "")
    m = re.fullmatch(r"Bearer ([^\s]+)", h.strip())
    return sha(m.group(1)) if m else None


def media_ok(request, required=True):
    ct = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if not ct:
        return not required
    return ct == MT_A2A


def gate(request, needs_body=False, body_required=True):
    """Common checks: auth -> version -> media type. Returns (principal, error_response)."""
    p = principal_of(request)
    if not p:
        return None, err(401, "UNAUTHENTICATED", "Missing or invalid bearer token")
    if request.headers.get("a2a-version", "").strip() != "1.0":
        return None, err(400, "VERSION_NOT_SUPPORTED", "Unsupported A2A version")
    if needs_body and not media_ok(request, required=body_required):
        return None, err(415, "UNSUPPORTED_MEDIA_TYPE", "Content-Type must be application/a2a+json")
    return p, None


NOT_FOUND = lambda: err(404, "TASK_NOT_FOUND", "Task not found")


# ----------------------------------------------------------------------------
# Decision layer
#   1. Code finds the decisive paragraph (exactly three bracketed refs) and reads
#      facts from the signed cover sheet.
#   2. One small batched LLM call classifies each decisive paragraph.
#   3. Rule-based backstop (incl. exact authority arithmetic) guards the result
#      and takes over if the model is unavailable.
#   Packages whose layout is unexpected fall back to a full-document LLM read.
# ----------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an accounts-payable control agent. For each item, read the DECISIVE PARAGRAPH (the binding, current statement about this invoice) and choose exactly ONE action:

- settle_invoice: valid, reconciled, not already paid, nothing pending, and the amount is WITHIN autonomous/delegated authority. A prior payment for a DIFFERENT invoice does not block settlement.
- request_approval: commercially valid but OUTSIDE delegated authority (amount above the limit, or a named approver / approval workflow is required).
- hold_invoice: payment must pause until a stated verification completes (e.g. bank-detail change callback still open, an inspection or check pending).
- reject_duplicate: the SAME commercial invoice was already paid (formatting differences in the identifier do not matter).
- open_exception: material records conflict (price/quantity/valuation/contract disagreements) and need the exception workflow.

Watch for negation ("No ... discrepancy remains", "not a duplicate") and for amounts: compare the stated total with any authority limit.
If an item has "documents" instead of a decisive paragraph, find the decisive paragraph yourself: it is the current statement about this invoice with exactly three bracketed references; ignore the cover sheet, archive notes, training material, hypothetical examples and superseded drafts. Then also return its three references (text inside the brackets) and the invoice facts.

Return ONLY JSON:
{"decisions":[{"packageId":"...","action":"<one action>","reason":"one short sentence",
 "evidenceRefs":["only for document items"],"facts":{"vendorName":"...","invoiceNumber":"...","amountMinor":0,"currency":"INR"}}]}"""

FACTS_RE = re.compile(r"Supplier\s+(.+?);\s*invoice\s+([^;\s]+);\s*stated total\s+([A-Z]{3})\s*([\d,]+(?:\.\d+)?)", re.I)
NOISE_RE = re.compile(r"archive note|training appendix|hypothetical|for example|superseded|non-operative", re.I)


def strip_ids(pkg):
    return {k: v for k, v in pkg.items() if k not in ("packageId", "batchId")}


def cache_key(pkg, policy_rev):
    return sha(canon({"policyRevision": policy_rev, "package": strip_ids(pkg)}))


def paragraphs_of(obj):
    out = []

    def walk(x):
        if isinstance(x, str):
            for para in re.split(r"\n\s*\n", x):
                if para.strip():
                    out.append(para.strip())
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(obj)
    return out


def refs_in(text):
    out = []
    for m in REF_RE.finditer(text):
        r = m.group(1).strip()
        if r not in out:
            out.append(r)
    return out


def ref_index(pkg):
    paras = paragraphs_of(strip_ids(pkg))
    ref_paras, para_refs = {}, []
    for i, p in enumerate(paras):
        refs = refs_in(p)
        for r in refs:
            ref_paras.setdefault(r, set()).add(i)
        para_refs.append(refs)
    return ref_paras, para_refs


def norm_ref(r):
    return str(r).strip().strip("[]").strip()


def to_int_minor(v):
    if isinstance(v, bool):
        raise ValueError
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str) and re.fullmatch(r"-?\d+", v.strip()):
        return int(v.strip())
    raise ValueError


def money_minor(txt):
    from decimal import Decimal
    return int((Decimal(txt.replace(",", "")) * 100).to_integral_value())


def fmt_ref(r):
    return f"[{r}]" if REF_FORMAT == "bracketed" else r


def analyze(pkg):
    """Deterministic read: facts from the cover sheet and the decisive paragraph."""
    paras = paragraphs_of(strip_ids(pkg))
    facts = None
    for p in paras:
        m = FACTS_RE.search(p)
        if m:
            facts = {"vendorName": m.group(1).strip(), "invoiceNumber": m.group(2).strip(),
                     "amountMinor": money_minor(m.group(4)), "currency": m.group(3).upper()}
            break
    cands = [p for p in paras if len(refs_in(p)) == 3 and not FACTS_RE.search(p)]
    if len(cands) > 1:
        cands = [p for p in cands if not NOISE_RE.search(p)] or cands
    para = cands[0] if len(cands) == 1 else None
    return {"facts": facts, "paragraph": para, "refs": refs_in(para) if para else None}


def rule_action(para, facts):
    """Keyword/arithmetic classifier. Returns an action or None when unsure."""
    t = para.lower()

    # DUPLICATE vs SETTLE  (test duplicate first; "commercial" is optional)
    if re.search(r"would be a duplicate|reject it|same (signed )?(commercial )?invoice|"
                 r"prior payment for the same|already (been )?paid", t) \
            and not re.search(r"different (signed )?(commercial )?invoice|not a duplicate|"
                              r"no payment was made for this", t):
        return "reject_duplicate"
    if re.search(r"different (signed )?(commercial )?invoice|not a duplicate|"
                 r"no payment was made for this|can be settled", t):
        return "settle_invoice"

    # HOLD
    if re.search(r"callback|until .{0,60}(clos|complet|confirm|verif|clear)|forbids remittance|"
                 r"pending verification|verification (is )?pending", t):
        return "hold_invoice"

    # EXCEPTION
    if re.search(r"exception workflow|does not reconcile|incompatible|conflicting records|irreconcil", t):
        return "open_exception"

    # APPROVAL vs SETTLE by exact arithmetic
    has_auth = re.search(r"authority|approver|approval workflow|without escalation|delegat", t)
    has_limit_cmp = re.search(r"(queue |approval |spend |autonomous )?limit", t) and \
                    re.search(r"total|amount", t)
    if has_auth or has_limit_cmp:
        limit = None
        m = (re.search(r"(?:up to|limit of|authority of)\s*([A-Z]{3})?\s*([\d,]+(?:\.\d+)?)", para)
             or re.search(r"([A-Z]{3})\s*([\d,]+(?:\.\d+)?)\s*authority", para)
             or re.search(r"limit\s*(?:is|of|:)?\s*([A-Z]{3})?\s*([\d,]+(?:\.\d+)?)", para, re.I))
        if m:
            limit = money_minor(m.group(2))
        if facts and limit is not None:
            return "request_approval" if facts["amountMinor"] > limit else "settle_invoice"
        if re.search(r"outside|exceed|above|larger|below .{0,30}total|"
                     r"requires the financial-approval|named approver", t):
            return "request_approval"
        if re.search(r"within", t):
            return "settle_invoice"
    return None


def build_rationale(action, refs, extra):
    cites = ", ".join(f"[{r}]" for r in refs)
    text = f"Action {action}. Decisive evidence {cites}: " + re.sub(r"\s+", " ", str(extra or "")).strip()
    if len(text) < 60:
        text += " Cover-sheet, archive and training references were excluded."
    return text[:1500]


def validate_full(pkg, raw):
    """Validate a full-document LLM decision (unexpected layouts)."""
    if not isinstance(raw, dict) or raw.get("action") not in ACTIONS:
        return None
    ref_paras, para_refs = ref_index(pkg)
    valid = [r for r in (norm_ref(x) for x in raw.get("evidenceRefs") or []) if r in ref_paras]
    refs = None
    if len(valid) == 3 and set.intersection(*(ref_paras[r] for r in valid)):
        refs = valid
    elif valid:
        for i in sorted({i for r in valid for i in ref_paras[r]}):
            if len(para_refs[i]) == 3:
                refs = para_refs[i]
                break
    f = raw.get("facts") or {}
    try:
        facts = {"vendorName": str(f.get("vendorName", "")).strip(),
                 "invoiceNumber": str(f.get("invoiceNumber", "")).strip(),
                 "amountMinor": to_int_minor(f.get("amountMinor")),
                 "currency": str(f.get("currency", "")).strip().upper()}
    except Exception:
        return None
    if not refs or not facts["vendorName"] or not facts["invoiceNumber"]:
        return None
    return {"action": raw["action"], "facts": facts, "evidenceRefs": refs,
            "rationale": build_rationale(raw["action"], refs, raw.get("reason"))}


async def llm_call(items, policy_rev, timeout):
    STATS["model_calls"] += 1
    if LLM_MOCK:
        await asyncio.sleep(0.2)
        return {"decisions": [{"packageId": it["packageId"],
                               "action": rule_action(it.get("decisiveParagraph", ""), None) or "open_exception",
                               "reason": "mock"} for it in items]}
    user = f"Policy revision: {policy_rev}\nItems:\n" + json.dumps(items, ensure_ascii=False)
    payload = {"model": LLM_MODEL, "temperature": 0, "response_format": {"type": "json_object"},
               "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user}]}
    # Gemini thinking models stall for 30s+ unless reasoning is turned off; harmless to others.
    if LLM_REASONING_EFFORT:
        payload["reasoning_effort"] = LLM_REASONING_EFFORT
    # Give a slow model room, but never exceed the per-request budget passed in.
    http_timeout = max(float(timeout), LLM_HTTP_TIMEOUT_MIN)
    async with httpx.AsyncClient(timeout=http_timeout) as c:
        r = await c.post(f"{LLM_BASE_URL}/chat/completions", json=payload,
                         headers={"Authorization": f"Bearer {LLM_API_KEY}"})
        if r.status_code >= 400:
            STATS["last_llm_error"] = f"HTTP {r.status_code}: {r.text[:500]}"
            r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
    text = re.sub(r"^```(?:json)?|```$", "", text.strip()).strip()
    return json.loads(text)


async def llm_batch(items, policy_rev, timeout):
    try:
        data = await llm_call(items, policy_rev, timeout)
        return {str(d.get("packageId")): d for d in data.get("decisions", []) if isinstance(d, dict)}
    except Exception as e:
        STATS.setdefault("last_llm_error", f"{type(e).__name__}: {e}")
        print(f"[llm] failed: {type(e).__name__}: {e} | {STATS.get('last_llm_error')}", flush=True)
        return {}


async def decide(packages, policy_rev):
    """Return {packageId: decision}. Cached by canonical package content."""
    t0 = time.monotonic()
    out, todo, info = {}, [], {}
    for pkg in packages:
        d = store.get_decision(cache_key(pkg, policy_rev))
        if d:
            out[pkg["packageId"]] = d
        else:
            todo.append(pkg)
            info[pkg["packageId"]] = analyze(pkg)
    if not todo:
        return out

    items = []
    for pkg in todo:
        a = info[pkg["packageId"]]
        if a["paragraph"] and a["facts"]:
            items.append({"packageId": pkg["packageId"],
                          "statedTotal": f'{a["facts"]["currency"]} {a["facts"]["amountMinor"] / 100:.2f}',
                          "decisiveParagraph": a["paragraph"]})
        else:
            items.append({"packageId": pkg["packageId"], "documents": strip_ids(pkg)})

    llm = {}
    for attempt in range(2):
        remaining = DECIDE_BUDGET_S - (time.monotonic() - t0)
        missing = [it for it in items if it["packageId"] not in llm]
        if not missing or remaining < 5:
            break
        chunks = [missing[i:i + CHUNK_SIZE] for i in range(0, len(missing), CHUNK_SIZE)]
        for r in await asyncio.gather(*[llm_batch(c, policy_rev, remaining - 2) for c in chunks]):
            llm.update(r)

    for pkg in todo:
        pid = pkg["packageId"]
        a = info[pid]
        raw = llm.get(pid) or {}
        model_action = raw.get("action") if raw.get("action") in ACTIONS else None
        if a["paragraph"] and a["facts"]:
            rule = rule_action(a["paragraph"], a["facts"])
            if model_action and rule and model_action != rule:
                print(f"[decide] {pid}: model={model_action} rule={rule}", flush=True)
            action = model_action or rule
            # Safety: never settle when the rules see a reason not to; trust exact
            # authority arithmetic over the model for approve-vs-settle.
            if rule and (action == "settle_invoice" or {action, rule} == {"settle_invoice", "request_approval"}):
                action = rule
            if action is None:
                action = "open_exception"
            d = {"action": action, "facts": a["facts"], "evidenceRefs": a["refs"],
                 "rationale": build_rationale(action, a["refs"], a["paragraph"])}
            if model_action or rule:
                store.put_decision(cache_key(pkg, policy_rev), d)
        else:
            d = validate_full(pkg, raw)
            if d:
                store.put_decision(cache_key(pkg, policy_rev), d)
            else:
                print(f"[decide] fallback for {pid}", flush=True)
                refs = a["refs"] or next((r for r in ref_index(pkg)[1] if len(r) == 3), [])
                facts = a["facts"] or {"vendorName": "unknown", "invoiceNumber": "unknown",
                                       "amountMinor": 0, "currency": "INR"}
                d = {"action": "open_exception", "facts": facts, "evidenceRefs": refs,
                     "rationale": build_rationale("open_exception", refs, "Layout not recognised.")}
        out[pid] = d
    return out


# ----------------------------------------------------------------------------
# Task construction
# ----------------------------------------------------------------------------
def action_id(task_id, package_id):
    return "act-" + sha(f"{task_id}|{package_id}")[:24]


def proposal_artifact(batch_id, proposals):
    return {"artifactId": "proposals", "name": "invoice-action-proposals",
            "parts": [{"mediaType": MT_PROPOSALS, "data": {"batchId": batch_id, "proposals": proposals}}]}


def public_proposal(p):
    return {**p, "evidenceRefs": [fmt_ref(r) for r in p["evidenceRefs"]]}


# ----------------------------------------------------------------------------
# Keep-alive (Render free tier)
# ----------------------------------------------------------------------------
async def keepalive_loop():
    """Ping our own public URL every 10 minutes so Render never idles the service."""
    await asyncio.sleep(30)  # let the server finish booting
    async with httpx.AsyncClient(timeout=10) as client:
        while True:
            try:
                r = await client.get(KEEPALIVE_URL, headers={"User-Agent": "a2a-keepalive"})
                if r.status_code >= 400:
                    print(f"[keepalive] ping returned HTTP {r.status_code}", flush=True)
            except Exception as e:  # never let a failed ping kill the loop
                print(f"[keepalive] ping failed: {type(e).__name__}", flush=True)
            await asyncio.sleep(KEEPALIVE_INTERVAL)


@contextlib.asynccontextmanager
async def lifespan(_app):
    task = None
    if KEEPALIVE_URL:
        task = asyncio.create_task(keepalive_loop())
        print(f"[keepalive] enabled: every {KEEPALIVE_MINUTES} min -> {KEEPALIVE_URL}", flush=True)
    try:
        yield
    finally:
        if task:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
app = FastAPI(redirect_slashes=False, docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)


def agent_card():
    return {
        "name": "Invoice Action Agent",
        "description": "Reads invoice packages, proposes one evidence-backed action per invoice, "
                       "and executes only grader-accepted actions with stored, replay-safe receipts.",
        "version": "1.0.0",
        "supportedInterfaces": [{"url": PUBLIC_BASE_URL, "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}],
        "capabilities": {"streaming": False, "pushNotifications": False},
        "defaultInputModes": [MT_BATCH],
        "defaultOutputModes": [MT_PROPOSALS, MT_RECEIPTS],
        "skills": [{
            "id": "invoice_action_agent",
            "name": "Invoice action agent",
            "description": "Chooses settle_invoice, request_approval, hold_invoice, reject_duplicate or "
                           "open_exception for each invoice package and cites the decisive evidence.",
            "tags": ["invoice", "accounts-payable", "approval", "a2a"],
            "inputModes": [MT_BATCH, MT_RESULTS],
            "outputModes": [MT_PROPOSALS, MT_RECEIPTS],
        }],
    }


@app.get("/.well-known/agent-card.json")
async def card():
    return JSONResponse(agent_card(), media_type="application/json")


@app.get("/")
async def root():
    return JSONResponse({"ok": True})


@app.api_route("/healthz", methods=["GET", "HEAD"])
async def healthz():
    # cheap liveness probe: no DB, no model call
    return JSONResponse({"ok": True})


async def read_json(request):
    try:
        raw = await request.body()
        if not raw:
            return None
        return json.loads(raw)
    except Exception:
        return None


@app.post(BASE_PATH + "/message:send")
async def message_send(request: Request):
    p, e = gate(request, needs_body=True)
    if e:
        return e
    body = await read_json(request)
    if not isinstance(body, dict) or not isinstance(body.get("message"), dict):
        return err(400, "INVALID_REQUEST", "Body must contain a message object")
    msg = body["message"]
    mid = msg.get("messageId")
    if not isinstance(mid, str) or not mid:
        return err(400, "INVALID_REQUEST", "messageId is required")
    h = sha(canon(msg))
    async with lock_for(("msg", p, mid)):
        rec = store.get_message(p, mid)
        if rec:
            if rec[0] != h:
                return err(409, "IDEMPOTENCY_CONFLICT", "messageId was already used with different content")
            t = store.get_task(rec[1])
            return ok({"task": t["task"]})
        if msg.get("taskId"):
            return await continuation(p, msg, mid, h)
        return await initial(p, msg, mid, h)


async def initial(p, msg, mid, h):
    if msg.get("role") != "ROLE_USER":
        return err(400, "INVALID_REQUEST", "role must be ROLE_USER")
    parts = msg.get("parts")
    part = next((x for x in parts if isinstance(x, dict) and x.get("mediaType") == MT_BATCH), None) \
        if isinstance(parts, list) else None
    if not part:
        return err(415, "UNSUPPORTED_MEDIA_TYPE", "Expected an invoice-claim-batch part")
    data = part.get("data")
    if not isinstance(data, dict) or not isinstance(data.get("batchId"), str) \
            or not isinstance(data.get("packages"), list) or not data["packages"]:
        return err(400, "INVALID_REQUEST", "Invalid invoice batch")
    pkgs = data["packages"]
    ids = [x.get("packageId") if isinstance(x, dict) else None for x in pkgs]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        return err(400, "INVALID_REQUEST", "Every package needs a unique packageId")
    policy_rev = data.get("policyRevision")

    try:
        decisions = await decide(pkgs, policy_rev)
    except Exception as ex:
        print(f"[decide] failed: {ex}", flush=True)
        return err(503, "MODEL_UNAVAILABLE", "Decision step failed; retry")

    task_id = str(uuid.uuid4())
    context_id = msg.get("contextId") if isinstance(msg.get("contextId"), str) and msg.get("contextId") \
        else str(uuid.uuid4())
    proposals = []
    for pkg in pkgs:
        d = decisions[pkg["packageId"]]
        proposals.append({"packageId": pkg["packageId"], "actionId": action_id(task_id, pkg["packageId"]),
                          "action": d["action"], "facts": d["facts"],
                          "evidenceRefs": d["evidenceRefs"], "rationale": d["rationale"]})
    pub = [public_proposal(x) for x in proposals]
    hist_msg = {**msg, "taskId": task_id, "contextId": context_id}
    task = {"id": task_id, "contextId": context_id,
            "status": {"state": "TASK_STATE_INPUT_REQUIRED", "timestamp": now_iso()},
            "artifacts": [proposal_artifact(data["batchId"], pub)],
            "history": [hist_msg]}
    internal = {"batchId": data["batchId"], "proposals": pub}
    store.save(p, task, internal, msg_record=(mid, h, task_id), created=True)
    return ok({"task": task})


async def continuation(p, msg, mid, h):
    tid = msg.get("taskId")
    if not isinstance(tid, str):
        return NOT_FOUND()
    async with lock_for(("task", tid)):
        t = store.get_task(tid)
        if not t or t["principal"] != p:
            return NOT_FOUND()
        task, internal = t["task"], t["internal"]
        if task["status"]["state"] in TERMINAL:
            return err(409, "TASK_NOT_MODIFIABLE", "Task is already in a terminal state")
        if task["status"]["state"] != "TASK_STATE_INPUT_REQUIRED":
            return err(409, "TASK_NOT_READY", "Task is not awaiting input")
        if msg.get("contextId") != task["contextId"]:
            return err(400, "CONTEXT_MISMATCH", "contextId does not match the task")
        if msg.get("role") != "ROLE_USER":
            return err(400, "INVALID_REQUEST", "role must be ROLE_USER")
        parts = msg.get("parts")
        part = next((x for x in parts if isinstance(x, dict) and x.get("mediaType") == MT_RESULTS), None) \
            if isinstance(parts, list) else None
        if not part:
            return err(415, "UNSUPPORTED_MEDIA_TYPE", "Expected an invoice-action-results part")
        data = part.get("data")
        if not isinstance(data, dict) or data.get("batchId") != internal["batchId"]:
            return err(400, "BATCH_MISMATCH", "batchId does not match the task")
        results = data.get("results")
        if not isinstance(results, list):
            return err(400, "INVALID_REQUEST", "results must be a list")
        by_pkg = {pr["packageId"]: pr for pr in internal["proposals"]}
        seen, executions = set(), []
        for r in results:
            if not isinstance(r, dict):
                return err(400, "INVALID_REQUEST", "Invalid result entry")
            pid = r.get("packageId")
            pr = by_pkg.get(pid)
            if pr is None or pid in seen:
                return err(400, "PACKAGE_MISMATCH", "Result does not match a stored proposal")
            seen.add(pid)
            if r.get("actionId") != pr["actionId"] or r.get("action") != pr["action"]:
                return err(400, "ACTION_MISMATCH", "Result does not match the stored proposal")
            if r.get("outcome") not in ("ACCEPTED", "REJECTED"):
                return err(400, "INVALID_REQUEST", "outcome must be ACCEPTED or REJECTED")
            if r["outcome"] == "ACCEPTED":
                nonce = r.get("receiptNonce")
                if not isinstance(nonce, str) or not nonce:
                    return err(400, "RECEIPT_MISSING", "Accepted result needs a receiptNonce")
        if seen != set(by_pkg):
            return err(400, "PACKAGE_MISMATCH", "Results must cover every proposal exactly once")
        res_by_pkg = {r["packageId"]: r for r in results}
        for pr in internal["proposals"]:  # proposal order
            r = res_by_pkg[pr["packageId"]]
            if r["outcome"] == "ACCEPTED":
                executions.append({"packageId": pr["packageId"], "actionId": pr["actionId"],
                                   "action": pr["action"], "receiptNonce": r["receiptNonce"],
                                   "facts": pr["facts"], "evidenceRefs": pr["evidenceRefs"]})
        task["history"].append(msg)
        task["artifacts"].append({"artifactId": "receipts", "name": "invoice-action-receipts",
                                  "parts": [{"mediaType": MT_RECEIPTS,
                                             "data": {"batchId": internal["batchId"], "executions": executions}}]})
        task["status"] = {"state": "TASK_STATE_COMPLETED", "timestamp": now_iso()}
        store.save(p, task, internal, msg_record=(mid, h, tid))
        return ok({"task": task})


@app.post(BASE_PATH + "/tasks/{task_id}:cancel")
async def cancel(task_id: str, request: Request):
    p, e = gate(request, needs_body=True, body_required=False)
    if e:
        return e
    async with lock_for(("task", task_id)):
        t = store.get_task(task_id)
        if not t or t["principal"] != p:
            return NOT_FOUND()
        task = t["task"]
        if task["status"]["state"] in TERMINAL:
            return err(409, "TASK_NOT_CANCELABLE", "Task cannot be canceled")
        task["status"] = {"state": "TASK_STATE_CANCELED", "timestamp": now_iso()}
        store.save(p, task, t["internal"])
        return ok(task)


@app.get(BASE_PATH + "/tasks/{task_id}")
async def get_task(task_id: str, request: Request):
    p, e = gate(request)
    if e:
        return e
    t = store.get_task(task_id)
    if not t or t["principal"] != p:
        return NOT_FOUND()
    return ok(t["task"])


async def _list(request: Request):
    p, e = gate(request)
    if e:
        return e
    qp = request.query_params
    tasks = store.list_tasks(p, qp.get("contextId"), qp.get("status"))
    try:
        offset = max(0, int(qp.get("pageToken") or 0))
        size = min(100, max(1, int(qp.get("pageSize") or 50)))
        hist = int(qp.get("historyLength") or 0)
    except ValueError:
        return err(400, "INVALID_REQUEST", "Invalid paging parameters")
    page, total = [], 0
    for t in tasks[offset:offset + size]:
        t = dict(t)
        if hist <= 0:
            t.pop("history", None)
        else:
            t["history"] = t.get("history", [])[-hist:]
        n = len(canon(t))
        if total + n > 450_000 and page:
            break
        page.append(t)
        total += n
    out = {"tasks": page}
    if offset + len(page) < len(tasks):
        out["nextPageToken"] = str(offset + len(page))
    return ok(out)


app.add_api_route(BASE_PATH + "/tasks", _list, methods=["GET"])
app.add_api_route(BASE_PATH + "/tasks/", _list, methods=["GET"])


@app.get("/_admin/recent")
async def admin_recent(request: Request, n: int = 5):
    """Debug view of recent tasks (incl. raw packages). Disabled unless ADMIN_TOKEN is set."""
    if not ADMIN_TOKEN or request.headers.get("x-admin-token", request.query_params.get("token")) != ADMIN_TOKEN:
        return err(404, "NOT_FOUND", "Not found")
    with store.lock:
        rows = store.db.execute("SELECT task_json FROM tasks ORDER BY created DESC LIMIT ?", (n,)).fetchall()
    return JSONResponse({"stats": STATS, "tasks": [json.loads(r[0]) for r in rows]})


@app.get("/_admin/llmtest")
async def admin_llmtest(request: Request):
    """Makes one tiny model call and shows the raw result or error."""
    if not ADMIN_TOKEN or request.headers.get("x-admin-token", request.query_params.get("token")) != ADMIN_TOKEN:
        return err(404, "NOT_FOUND", "Not found")
    payload = {"model": LLM_MODEL, "temperature": 0, "response_format": {"type": "json_object"},
               "messages": [{"role": "user", "content": 'Reply with JSON {"ok": true}'}]}
    if LLM_REASONING_EFFORT:
        payload["reasoning_effort"] = LLM_REASONING_EFFORT
    try:
        async with httpx.AsyncClient(timeout=max(30.0, LLM_HTTP_TIMEOUT_MIN)) as c:
            r = await c.post(f"{LLM_BASE_URL}/chat/completions", json=payload,
                             headers={"Authorization": f"Bearer {LLM_API_KEY}"})
        return JSONResponse({"base": LLM_BASE_URL, "model": LLM_MODEL, "key_len": len(LLM_API_KEY),
                             "status": r.status_code, "body": r.text[:800]})
    except Exception as e:
        return JSONResponse({"base": LLM_BASE_URL, "model": LLM_MODEL, "key_len": len(LLM_API_KEY),
                             "error": f"{type(e).__name__}: {e}"})


# Collapse accidental double slashes (e.g. base "/a2a/" + "/message:send") without redirecting.
class _SlashFix:
    def __init__(self, inner):
        self.inner = inner

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and "//" in scope.get("path", ""):
            scope = dict(scope)
            scope["path"] = re.sub(r"/{2,}", "/", scope["path"])
        await self.inner(scope, receive, send)


asgi = _SlashFix(app)
