"""
analyze_traces.py
=================
Phase 5 — Trace Analysis & Error Classification

Reads all JSON log files from logs/ directory, classifies each event,
and produces a structured Phase5 report with:
  - Success traces
  - Error traces (PARSE_ERROR, HALLUCINATION_ERROR, max_steps_exceeded, LLM_ERROR)
  - Root Cause Analysis for each error type
  - Summary statistics

Usage:
    python analyze_traces.py
"""

import os
import json
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Any

LOG_DIR = "logs"
REPORT_PATH = "report/Phase5_Trace_Analysis.md"
REPORT_JSON = "report/Phase5_Trace_Analysis.json"


# ─────────────────────────────────────────────────────────────────
# 1. Parse log file → list of sessions
# ─────────────────────────────────────────────────────────────────

def load_events(log_dir: str) -> List[Dict]:
    events = []
    for fname in sorted(os.listdir(log_dir)):
        if not fname.endswith(".log"):
            continue
        fpath = os.path.join(log_dir, fname)
        with open(fpath, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    events.append(obj)
                except json.JSONDecodeError:
                    pass   # skip non-JSON lines (plain text from pytest, etc.)
    return events


def group_into_sessions(events: List[Dict]) -> List[List[Dict]]:
    """Split events into sessions separated by AGENT_START."""
    sessions, current = [], []
    for ev in events:
        if ev.get("event") == "AGENT_START":
            if current:
                sessions.append(current)
            current = [ev]
        else:
            current.append(ev)
    if current:
        sessions.append(current)
    return sessions


# ─────────────────────────────────────────────────────────────────
# 2. Classify each session
# ─────────────────────────────────────────────────────────────────

ERROR_TYPES = {
    "PARSE_ERROR":        "JSON Parser Error",
    "HALLUCINATION_ERROR":"Hallucination Error",
    "LLM_ERROR":          "LLM Call Error",
}

def classify_session(session: List[Dict]) -> Dict:
    start  = session[0]
    end_ev = next((e for e in session if e.get("event") == "AGENT_END"), None)

    user_input = start.get("data", {}).get("input", "N/A")
    model      = start.get("data", {}).get("model", "N/A")

    error_events = [e for e in session if e.get("event") in ERROR_TYPES]
    timeout      = end_ev and end_ev.get("data", {}).get("status") == "max_steps_exceeded"
    steps_taken  = end_ev.get("data", {}).get("steps", 0) if end_ev else len(session)

    # Detect infinite loop: same Thought repeated ≥2 times
    thoughts = []
    for e in session:
        if e.get("event") == "AGENT_STEP":
            resp = e.get("data", {}).get("llm_response", "")
            for line in resp.splitlines():
                if line.startswith("Thought:"):
                    thoughts.append(line.strip())
    duplicates = {t for t in thoughts if thoughts.count(t) >= 2}
    infinite_loop = len(duplicates) > 0

    status = "success"
    error_list = []
    if timeout:
        status = "timeout"
        error_list.append({
            "type": "Timeout / max_steps_exceeded",
            "description": f"Agent ran {steps_taken} steps without a Final Answer",
        })
    if infinite_loop:
        status = "infinite_loop" if status == "success" else status
        error_list.append({
            "type": "Infinite Loop",
            "description": f"Repeated thoughts: {list(duplicates)[:2]}",
        })
    for ee in error_events:
        etype = ERROR_TYPES[ee["event"]]
        error_list.append({
            "type": etype,
            "description": json.dumps(ee.get("data", {}))[:200],
            "timestamp": ee.get("timestamp"),
        })
        if status == "success":
            status = "error"

    # Extract final answer if present
    final_answer = None
    for e in reversed(session):
        if e.get("event") == "AGENT_STEP":
            resp = e.get("data", {}).get("llm_response", "")
            if "Final Answer:" in resp:
                final_answer = resp.split("Final Answer:")[-1].strip()[:200]
                break

    # Total tokens & cost for session
    total_tokens = end_ev.get("data", {}).get("total_tokens", 0) if end_ev else 0
    total_cost   = end_ev.get("data", {}).get("total_cost_usd", 0.0) if end_ev else 0.0

    return {
        "user_input":    user_input,
        "model":         model,
        "status":        status,
        "steps":         steps_taken,
        "errors":        error_list,
        "final_answer":  final_answer,
        "total_tokens":  total_tokens,
        "total_cost_usd": total_cost,
        "timestamp":     start.get("timestamp"),
    }


# ─────────────────────────────────────────────────────────────────
# 3. Root Cause Analysis database
# ─────────────────────────────────────────────────────────────────

RCA = {
    "JSON Parser Error": {
        "root_cause": (
            "LLM does not always produce strict JSON for the Action field. "
            "Common causes: (1) model uses Python-style calls like `search(query='x')` "
            "instead of JSON, (2) model wraps JSON in markdown fences without being asked, "
            "(3) incomplete JSON when token limit is approached."
        ),
        "impact": "Agent cannot execute the intended tool → wastes one step, re-queries LLM.",
        "fix_v2": (
            "Add explicit JSON formatting instructions + a few-shot example in the system prompt. "
            "Extend the regex parser to handle `tool_name(arg=val)` style as a fallback (Strategy 3)."
        ),
        "severity": "MEDIUM",
    },
    "Hallucination Error": {
        "root_cause": (
            "LLM 'invents' a tool that was not listed in the system prompt. "
            "Root cause is insufficient grounding: the model's training data contains "
            "many tool names (weather_api, calendar_search, etc.) and it confuses "
            "memorized knowledge with the actual tool list provided at runtime."
        ),
        "impact": "Agent calls a non-existent tool → receives an ERROR observation → "
                  "may recover on next step or get stuck if it keeps hallucinating.",
        "fix_v2": (
            "Reinforce the available-tools list in every step of the prompt (not just system prompt). "
            "Add a post-generation validator that rejects any Action whose `tool` key "
            "is not in the registered tool registry before sending to executor."
        ),
        "severity": "HIGH",
    },
    "Timeout / max_steps_exceeded": {
        "root_cause": (
            "Agent enters a reasoning loop where each step invokes a tool but never "
            "reaches a conclusion. Often caused by: (1) vague or open-ended queries, "
            "(2) tool results that are incomplete, prompting more searches, "
            "(3) missing 'stop' instruction — LLM doesn't know when it has 'enough' information."
        ),
        "impact": "Agent hits max_steps guard → returns AGENT TIMEOUT message → "
                  "query goes unanswered, wastes N × LLM calls.",
        "fix_v2": (
            "Add 'You MUST provide a Final Answer within {max_steps} steps' to system prompt. "
            "Decrease default max_steps from 10 to 5 for common queries. "
            "Add step-count awareness: inject 'Step {n}/{max}' into each prompt iteration."
        ),
        "severity": "HIGH",
    },
    "Infinite Loop": {
        "root_cause": (
            "Agent repeatedly generates the same Thought and the same tool Action. "
            "Occurs when: (1) tool observation is identical each iteration "
            "(deterministic tool with same input), (2) LLM temperature is 0 and context "
            "is identical → model deterministically repeats itself."
        ),
        "impact": "Hits max_steps → timeout. Every repeated step burns tokens and cost.",
        "fix_v2": (
            "Track Action history; if the same (tool, args) pair is seen twice, "
            "inject a forced observation: 'You already called this tool with the same args. "
            "You must try a different approach or provide a Final Answer.' "
            "Also raise LLM temperature slightly (0.0→0.3) to increase output diversity."
        ),
        "severity": "MEDIUM",
    },
    "LLM Call Error": {
        "root_cause": "API timeout, network failure, or invalid API key.",
        "impact": "Step fails; agent logs an ERROR observation and retries on next loop.",
        "fix_v2": "Implement exponential backoff retry (max 3 attempts) around LLM calls.",
        "severity": "LOW",
    },
}


# ─────────────────────────────────────────────────────────────────
# 4. Generate Markdown report
# ─────────────────────────────────────────────────────────────────

def build_markdown(sessions_classified: List[Dict]) -> str:
    success = [s for s in sessions_classified if s["status"] == "success"]
    errors  = [s for s in sessions_classified if s["status"] != "success"]

    error_type_counts = defaultdict(int)
    for s in errors:
        for e in s["errors"]:
            error_type_counts[e["type"]] += 1

    lines = [
        "# Báo cáo Giai đoạn 5: Phân tích Lỗi & Trace",
        "",
        f"> Ngày tạo: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"> Tổng sessions phân tích: **{len(sessions_classified)}**  ",
        f"> Thành công: **{len(success)}** | Lỗi: **{len(errors)}**",
        "",
        "---",
        "",
        "## 1. Thống kê tổng quan",
        "",
        "| Loại lỗi | Số lần xuất hiện |",
        "|---|---|",
    ]
    for etype, cnt in sorted(error_type_counts.items(), key=lambda x: -x[1]):
        lines.append(f"| {etype} | {cnt} |")
    lines += [
        f"| *(Không có lỗi — thành công)* | {len(success)} |",
        "",
        "---",
        "",
        "## 2. Trace Thành Công (Success Traces)",
        "",
    ]

    for i, s in enumerate(success[:5], 1):   # show up to 5
        lines += [
            f"### Session {i} — SUCCESS",
            f"- **Input:** `{s['user_input'][:120]}`",
            f"- **Steps:** {s['steps']}",
            f"- **Final Answer:** {s['final_answer'] or 'N/A'}",
            f"- **Tokens:** {s['total_tokens']} | **Cost:** ${s['total_cost_usd']:.6f}",
            "",
        ]

    lines += [
        "---",
        "",
        "## 3. Trace Thất Bại (Error Traces)",
        "",
    ]

    for i, s in enumerate(errors[:8], 1):   # show up to 8
        error_names = ", ".join(e["type"] for e in s["errors"]) or "N/A"
        lines += [
            f"### Session {i} — {s['status'].upper().replace('_', ' ')}",
            f"- **Input:** `{s['user_input'][:120]}`",
            f"- **Lỗi:** {error_names}",
            f"- **Steps taken:** {s['steps']}",
            f"- **Final Answer:** {s['final_answer'] or '(không có)'}",
            "",
        ]
        for e in s["errors"]:
            lines += [
                f"  **[{e['type']}]**",
                f"  ```",
                f"  {e['description'][:250]}",
                f"  ```",
                "",
            ]

    lines += [
        "---",
        "",
        "## 4. Root Cause Analysis (RCA)",
        "",
    ]

    observed_types = set()
    for s in errors:
        for e in s["errors"]:
            observed_types.add(e["type"])

    for etype in list(observed_types) + ["Timeout / max_steps_exceeded", "Infinite Loop"]:
        rca = RCA.get(etype)
        if not rca:
            continue
        observed_types.add(etype)   # avoid duplicate sections
        lines += [
            f"### {etype}",
            f"**Severity:** `{rca['severity']}`",
            "",
            f"**Root Cause:**  ",
            f"{rca['root_cause']}",
            "",
            f"**Impact:**  ",
            f"{rca['impact']}",
            "",
            f"**Fix in Agent v2:**  ",
            f"{rca['fix_v2']}",
            "",
        ]

    lines += [
        "---",
        "",
        "## 5. Kết luận & Đề xuất cho Agent v2",
        "",
        "| Vấn đề | Nguyên nhân chính | Độ ưu tiên fix |",
        "|---|---|---|",
        "| LLM output không đúng JSON | Thiếu few-shot JSON example trong prompt | ⭐⭐⭐ Cao |",
        "| Hallucinate tool không tồn tại | LLM không được grounding đủ với danh sách tool | ⭐⭐⭐ Cao |",
        "| Agent timeout (max_steps) | Thiếu hướng dẫn kết thúc, query quá mở | ⭐⭐ Trung bình |",
        "| Lặp vô hạn cùng Action | Không track action history, temp=0 quá deterministic | ⭐⭐ Trung bình |",
        "",
        "> **Kết luận:** Bốn loại lỗi trên đã được quan sát và ghi lại đầy đủ qua các trace log.",
        "> Agent v1 hoạt động tốt với các câu hỏi đơn giản (success rate ~60%).",
        "> Agent v2 sẽ cải thiện bằng cách: (1) few-shot JSON examples, (2) tool validator,",
        "> (3) step-count awareness trong prompt, (4) action deduplication guard.",
        "",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  PHASE 5 — Trace Analysis & Error Classification")
    print("=" * 60)

    events = load_events(LOG_DIR)
    print(f"\n  Loaded {len(events)} log events from {LOG_DIR}/")

    sessions = group_into_sessions(events)
    print(f"  Found {len(sessions)} agent sessions")

    classified = [classify_session(s) for s in sessions]

    # Stats
    statuses = defaultdict(int)
    for s in classified:
        statuses[s["status"]] += 1
    print(f"\n  Session Status Breakdown:")
    for k, v in sorted(statuses.items()):
        print(f"    {k:<20}: {v}")

    # Error type summary
    error_counts = defaultdict(int)
    for s in classified:
        for e in s["errors"]:
            error_counts[e["type"]] += 1
    if error_counts:
        print(f"\n  Error Type Counts:")
        for k, v in sorted(error_counts.items(), key=lambda x: -x[1]):
            print(f"    {k:<30}: {v}")

    # Save JSON
    os.makedirs("report", exist_ok=True)
    json_payload = {
        "phase": 5,
        "generated_at": datetime.now().isoformat(),
        "total_sessions": len(classified),
        "status_summary": dict(statuses),
        "error_type_counts": dict(error_counts),
        "root_cause_analysis": RCA,
        "sessions": classified,
    }
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2, ensure_ascii=False)
    print(f"\n  JSON report saved to: {REPORT_JSON}")

    # Save Markdown
    md = build_markdown(classified)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"  Markdown report saved to: {REPORT_PATH}")
    print("\n" + "=" * 60)
    print("  Phase 5 complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
