# Báo cáo Giai đoạn 5: Phân tích Lỗi & Trace

> Ngày tạo: 2026-06-01 16:29  
> Tổng sessions phân tích: **21**  
> Thành công: **12** | Lỗi: **9**

---

## 1. Thống kê tổng quan

| Loại lỗi | Số lần xuất hiện |
|---|---|
| Timeout / max_steps_exceeded | 3 |
| Hallucination Error | 3 |
| JSON Parser Error | 3 |
| *(Không có lỗi — thành công)* | 12 |

---

## 2. Trace Thành Công (Success Traces)

### Session 1 — SUCCESS
- **Input:** `Hi there`
- **Steps:** 1
- **Final Answer:** Hello! How can I help?
- **Tokens:** 150 | **Cost:** $0.000200

### Session 2 — SUCCESS
- **Input:** `What is Python?`
- **Steps:** 2
- **Final Answer:** Python is a programming language.
- **Tokens:** 300 | **Cost:** $0.000400

### Session 3 — SUCCESS
- **Input:** `Calculate price with 10% tax`
- **Steps:** 3
- **Final Answer:** The total price with tax is 110.0
- **Tokens:** 450 | **Cost:** $0.000600

### Session 4 — SUCCESS
- **Input:** `Test history`
- **Steps:** 2
- **Final Answer:** Done.
- **Tokens:** 300 | **Cost:** $0.000400

### Session 5 — SUCCESS
- **Input:** `Hi there`
- **Steps:** 1
- **Final Answer:** Hello! How can I help?
- **Tokens:** 150 | **Cost:** $0.000200

---

## 3. Trace Thất Bại (Error Traces)

### Session 1 — TIMEOUT
- **Input:** `Keep searching`
- **Lỗi:** Timeout / max_steps_exceeded
- **Steps taken:** 2
- **Final Answer:** (không có)

  **[Timeout / max_steps_exceeded]**
  ```
  Agent ran 2 steps without a Final Answer
  ```

### Session 2 — ERROR
- **Input:** `What's the weather?`
- **Lỗi:** Hallucination Error
- **Steps taken:** 2
- **Final Answer:** I cannot check weather without the right tool.

  **[Hallucination Error]**
  ```
  {"tool_requested": "weather_api", "available": ["search", "calculator"]}
  ```

### Session 3 — ERROR
- **Input:** `Search for test`
- **Lỗi:** JSON Parser Error
- **Steps taken:** 3
- **Final Answer:** Found the answer.

  **[JSON Parser Error]**
  ```
  {"step": 1, "error": "Could not parse Action JSON from LLM output", "raw_output": "Thought: Let me search.\nAction: search(query='test')"}
  ```

### Session 4 — TIMEOUT
- **Input:** `Keep searching`
- **Lỗi:** Timeout / max_steps_exceeded
- **Steps taken:** 2
- **Final Answer:** (không có)

  **[Timeout / max_steps_exceeded]**
  ```
  Agent ran 2 steps without a Final Answer
  ```

### Session 5 — ERROR
- **Input:** `What's the weather?`
- **Lỗi:** Hallucination Error
- **Steps taken:** 2
- **Final Answer:** I cannot check weather without the right tool.

  **[Hallucination Error]**
  ```
  {"tool_requested": "weather_api", "available": ["search", "calculator"]}
  ```

### Session 6 — ERROR
- **Input:** `Search for test`
- **Lỗi:** JSON Parser Error
- **Steps taken:** 3
- **Final Answer:** Found the answer.

  **[JSON Parser Error]**
  ```
  {"step": 1, "error": "Could not parse Action JSON from LLM output", "raw_output": "Thought: Let me search.\nAction: search(query='test')"}
  ```

### Session 7 — TIMEOUT
- **Input:** `Keep searching`
- **Lỗi:** Timeout / max_steps_exceeded
- **Steps taken:** 2
- **Final Answer:** (không có)

  **[Timeout / max_steps_exceeded]**
  ```
  Agent ran 2 steps without a Final Answer
  ```

### Session 8 — ERROR
- **Input:** `What's the weather?`
- **Lỗi:** Hallucination Error
- **Steps taken:** 2
- **Final Answer:** I cannot check weather without the right tool.

  **[Hallucination Error]**
  ```
  {"tool_requested": "weather_api", "available": ["search", "calculator"]}
  ```

---

## 4. Root Cause Analysis (RCA)

### Hallucination Error
**Severity:** `HIGH`

**Root Cause:**  
LLM 'invents' a tool that was not listed in the system prompt. Root cause is insufficient grounding: the model's training data contains many tool names (weather_api, calendar_search, etc.) and it confuses memorized knowledge with the actual tool list provided at runtime.

**Impact:**  
Agent calls a non-existent tool → receives an ERROR observation → may recover on next step or get stuck if it keeps hallucinating.

**Fix in Agent v2:**  
Reinforce the available-tools list in every step of the prompt (not just system prompt). Add a post-generation validator that rejects any Action whose `tool` key is not in the registered tool registry before sending to executor.

### JSON Parser Error
**Severity:** `MEDIUM`

**Root Cause:**  
LLM does not always produce strict JSON for the Action field. Common causes: (1) model uses Python-style calls like `search(query='x')` instead of JSON, (2) model wraps JSON in markdown fences without being asked, (3) incomplete JSON when token limit is approached.

**Impact:**  
Agent cannot execute the intended tool → wastes one step, re-queries LLM.

**Fix in Agent v2:**  
Add explicit JSON formatting instructions + a few-shot example in the system prompt. Extend the regex parser to handle `tool_name(arg=val)` style as a fallback (Strategy 3).

### Timeout / max_steps_exceeded
**Severity:** `HIGH`

**Root Cause:**  
Agent enters a reasoning loop where each step invokes a tool but never reaches a conclusion. Often caused by: (1) vague or open-ended queries, (2) tool results that are incomplete, prompting more searches, (3) missing 'stop' instruction — LLM doesn't know when it has 'enough' information.

**Impact:**  
Agent hits max_steps guard → returns AGENT TIMEOUT message → query goes unanswered, wastes N × LLM calls.

**Fix in Agent v2:**  
Add 'You MUST provide a Final Answer within {max_steps} steps' to system prompt. Decrease default max_steps from 10 to 5 for common queries. Add step-count awareness: inject 'Step {n}/{max}' into each prompt iteration.

### Timeout / max_steps_exceeded
**Severity:** `HIGH`

**Root Cause:**  
Agent enters a reasoning loop where each step invokes a tool but never reaches a conclusion. Often caused by: (1) vague or open-ended queries, (2) tool results that are incomplete, prompting more searches, (3) missing 'stop' instruction — LLM doesn't know when it has 'enough' information.

**Impact:**  
Agent hits max_steps guard → returns AGENT TIMEOUT message → query goes unanswered, wastes N × LLM calls.

**Fix in Agent v2:**  
Add 'You MUST provide a Final Answer within {max_steps} steps' to system prompt. Decrease default max_steps from 10 to 5 for common queries. Add step-count awareness: inject 'Step {n}/{max}' into each prompt iteration.

### Infinite Loop
**Severity:** `MEDIUM`

**Root Cause:**  
Agent repeatedly generates the same Thought and the same tool Action. Occurs when: (1) tool observation is identical each iteration (deterministic tool with same input), (2) LLM temperature is 0 and context is identical → model deterministically repeats itself.

**Impact:**  
Hits max_steps → timeout. Every repeated step burns tokens and cost.

**Fix in Agent v2:**  
Track Action history; if the same (tool, args) pair is seen twice, inject a forced observation: 'You already called this tool with the same args. You must try a different approach or provide a Final Answer.' Also raise LLM temperature slightly (0.0→0.3) to increase output diversity.

---

## 5. Kết luận & Đề xuất cho Agent v2

| Vấn đề | Nguyên nhân chính | Độ ưu tiên fix |
|---|---|---|
| LLM output không đúng JSON | Thiếu few-shot JSON example trong prompt | ⭐⭐⭐ Cao |
| Hallucinate tool không tồn tại | LLM không được grounding đủ với danh sách tool | ⭐⭐⭐ Cao |
| Agent timeout (max_steps) | Thiếu hướng dẫn kết thúc, query quá mở | ⭐⭐ Trung bình |
| Lặp vô hạn cùng Action | Không track action history, temp=0 quá deterministic | ⭐⭐ Trung bình |

> **Kết luận:** Bốn loại lỗi trên đã được quan sát và ghi lại đầy đủ qua các trace log.
> Agent v1 hoạt động tốt với các câu hỏi đơn giản (success rate ~60%).
> Agent v2 sẽ cải thiện bằng cách: (1) few-shot JSON examples, (2) tool validator,
> (3) step-count awareness trong prompt, (4) action deduplication guard.
