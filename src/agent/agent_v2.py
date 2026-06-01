"""
agent_v2.py — ReAct Agent v2
==============================
Improvements over v1 based on Phase 5 Root Cause Analysis:

Fix 1 (JSON Parser Error)    → Few-shot JSON example + step-count awareness in system prompt
Fix 2 (Hallucination Error)  → Pre-execution tool validator rejects hallucinated tool calls
Fix 3 (Timeout)              → Step-count awareness + "Final Answer by step N" instruction
Fix 4 (Infinite Loop)        → Action deduplication guard (same tool+args detected & blocked)
Fix 5 (LLM Error)            → Exponential backoff retry (up to 3 attempts) on LLM call failure
"""

import re
import json
import time
from typing import List, Dict, Any, Optional, Callable
from src.core.llm_provider import LLMProvider
from src.telemetry.logger import logger
from src.telemetry.metrics import tracker


class ReActAgentV2:
    """
    ReAct Agent v2 — Improved over v1 with:
    - Richer system prompt (few-shot JSON example, step-count awareness)
    - Pre-execution tool validator (hallucination guard)
    - Action deduplication guard (infinite loop prevention)
    - LLM retry with exponential backoff
    """

    VERSION = "v2"

    def __init__(
        self,
        llm: LLMProvider,
        tools: List[Dict[str, Any]],
        max_steps: int = 5,
        max_retries: int = 3,
    ):
        self.llm = llm
        self.tools = {t["name"]: t for t in tools}
        self.tools_list = tools
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.history: List[str] = []
        self._action_history: List[str] = []  # FIX 4: deduplication store

    # ──────────────────────────────────────────────
    # FIX 1 & 3: Improved system prompt
    # ──────────────────────────────────────────────

    def get_system_prompt(self, current_step: int = 1) -> str:
        """
        v2 improvements:
        - Explicit few-shot JSON example so LLM learns exact format
        - Step counter injected so LLM knows urgency
        - Explicit rule: "no tool not in list"
        - Explicit deadline: "provide Final Answer by step max_steps"
        """
        tool_descriptions = "\n".join(
            f'  - {t["name"]}: {t["description"]}' for t in self.tools_list
        )
        steps_left = self.max_steps - current_step + 1

        return (
            "You are a precise AI assistant that solves tasks step by step using tools.\n\n"
            f"Available tools:\n{tool_descriptions}\n\n"
            "=== FORMAT (follow EXACTLY) ===\n"
            "Thought: <your concise reasoning — 1-2 sentences>\n"
            'Action: {"tool": "<tool_name>", "args": {"<key>": "<value>"}}\n'
            "Observation: <system fills this in — DO NOT write it yourself>\n\n"
            "=== FEW-SHOT EXAMPLE ===\n"
            "User: What is 10% of 500?\n"
            "Thought: I need to calculate 10% of 500.\n"
            'Action: {"tool": "calculator", "args": {"expression": "500 * 0.10"}}\n'
            "Observation: 50.0\n"
            "Thought: I have the result.\n"
            "Final Answer: 10% of 500 is 50.0\n\n"
            "=== RULES ===\n"
            "1. Output EXACTLY ONE Action per turn as raw JSON (no markdown fences).\n"
            "2. ONLY use tool names from the list above — never invent new ones.\n"
            "3. Do NOT write Observation yourself — wait for the system.\n"
            "4. When you have the answer, write:\n"
            "   Thought: I now have all information.\n"
            "   Final Answer: <complete answer>\n"
            f"5. You are on step {current_step} of {self.max_steps}. "
            f"You have {steps_left} step(s) remaining. "
            f"{'PROVIDE FINAL ANSWER NOW — this is your last step!' if steps_left <= 1 else 'Be efficient.'}\n"
        )

    # ──────────────────────────────────────────────
    # Core ReAct Loop
    # ──────────────────────────────────────────────

    def run(self, user_input: str) -> str:
        logger.log_event("AGENT_START", {
            "version": self.VERSION,
            "input": user_input,
            "model": self.llm.model_name,
            "max_steps": self.max_steps,
        })

        self.history = []
        self._action_history = []
        steps = 0
        total_tokens = 0
        total_cost = 0.0

        while steps < self.max_steps:
            steps += 1

            # ── 1. Build prompt ──
            prompt = self._build_prompt(user_input)

            # ── 2. Call LLM with retry (FIX 5) ──
            result = self._call_llm_with_retry(prompt, steps)
            if result is None:
                self.history.append("Observation: [ERROR] LLM unavailable after retries. Skipping step.")
                continue

            content     = result.get("content", "")
            usage       = result.get("usage", {})
            latency_ms  = result.get("latency_ms", 0)
            provider    = result.get("provider", "unknown")

            tracker.track_request(provider, self.llm.model_name, usage, latency_ms)
            total_tokens += usage.get("total_tokens", 0)
            total_cost   += tracker.calculate_cost(self.llm.model_name, usage)

            logger.log_event("AGENT_STEP", {
                "version": self.VERSION,
                "step": steps,
                "llm_response": content[:500],
                "latency_ms": latency_ms,
                "tokens": usage,
            })

            # ── 3. Check Final Answer ──
            final = self._parse_final_answer(content)
            if final is not None:
                logger.log_event("AGENT_END", {
                    "version": self.VERSION, "steps": steps,
                    "status": "final_answer",
                    "total_tokens": total_tokens,
                    "total_cost_usd": round(total_cost, 6),
                })
                return final

            thought     = self._parse_thought(content)
            action_json = self._parse_action(content)

            # ── 4a. Parse error ──
            if action_json is None:
                logger.log_event("PARSE_ERROR", {
                    "version": self.VERSION, "step": steps,
                    "raw_output": content[:300],
                })
                self.history.append(content.strip())
                self.history.append(
                    "Observation: [SYSTEM] Your response did not contain a valid Action JSON. "
                    "Follow the format: Action: {\"tool\": \"name\", \"args\": {\"key\": \"value\"}}"
                )
                continue

            tool_name = action_json.get("tool", "")
            tool_args = action_json.get("args", {})

            # ── 4b. Hallucination guard (FIX 2) ──
            if tool_name not in self.tools:
                logger.log_event("HALLUCINATION_ERROR", {
                    "version": self.VERSION, "step": steps,
                    "tool_requested": tool_name,
                    "available": list(self.tools.keys()),
                })
                self.history.append(
                    f"Thought: {thought}\n"
                    f'Action: {{"tool": "{tool_name}", "args": {json.dumps(tool_args)}}}\n'
                    f"Observation: [ERROR] Tool '{tool_name}' does not exist. "
                    f"You MUST only use these tools: {list(self.tools.keys())}. Try a different approach."
                )
                continue

            # ── 4c. Deduplication guard (FIX 4) ──
            action_key = f"{tool_name}:{json.dumps(tool_args, sort_keys=True)}"
            if action_key in self._action_history:
                logger.log_event("DUPLICATE_ACTION", {
                    "version": self.VERSION, "step": steps,
                    "action_key": action_key,
                })
                self.history.append(
                    f"Thought: {thought}\n"
                    f"Observation: [SYSTEM] You already called '{tool_name}' with the same arguments. "
                    "The result won't change. Use a different tool or provide a Final Answer."
                )
                continue
            self._action_history.append(action_key)

            # ── 5. Execute tool ──
            observation = self._execute_tool(tool_name, tool_args)
            logger.log_event("TOOL_CALL", {
                "version": self.VERSION, "step": steps,
                "tool": tool_name, "args": tool_args,
                "observation": str(observation)[:300],
            })

            # ── 6. Append to history ──
            self.history.append(
                f"Thought: {thought}\n"
                f'Action: {json.dumps({"tool": tool_name, "args": tool_args})}\n'
                f"Observation: {observation}"
            )

        # Exhausted
        logger.log_event("AGENT_END", {
            "version": self.VERSION, "steps": steps,
            "status": "max_steps_exceeded",
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 6),
        })
        return (
            f"[AGENT TIMEOUT] Reached {self.max_steps} steps without a final answer. "
            + "\n".join(self.history[-2:])
        )

    # ──────────────────────────────────────────────
    # FIX 5: LLM call with exponential backoff
    # ──────────────────────────────────────────────

    def _call_llm_with_retry(self, prompt: str, current_step: int) -> Optional[Dict]:
        delay = 1.0
        for attempt in range(1, self.max_retries + 1):
            try:
                result = self.llm.generate(
                    prompt,
                    system_prompt=self.get_system_prompt(current_step)
                )
                return result
            except Exception as e:
                logger.log_event("LLM_ERROR", {
                    "version": self.VERSION,
                    "step": current_step,
                    "attempt": attempt,
                    "error": str(e),
                    "retry_in_s": delay if attempt < self.max_retries else None,
                })
                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay *= 2   # exponential backoff
        return None

    # ──────────────────────────────────────────────
    # Prompt Builder — same as v1 + step hint
    # ──────────────────────────────────────────────

    def _build_prompt(self, user_input: str) -> str:
        parts = [f"User Query: {user_input}"]
        if self.history:
            parts.append("\n--- Previous Steps ---")
            parts.extend(self.history)
            parts.append("--- End Previous Steps ---\n")
            parts.append("Continue reasoning from where you left off.")
        return "\n".join(parts)

    # ──────────────────────────────────────────────
    # Parsing Helpers (same 3-strategy as v1)
    # ──────────────────────────────────────────────

    def _parse_final_answer(self, text: str) -> Optional[str]:
        m = re.search(r"Final\s*Answer\s*:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
        return m.group(1).strip() if m else None

    def _parse_thought(self, text: str) -> str:
        m = re.search(
            r"Thought\s*:\s*(.+?)(?=Action\s*:|Final\s*Answer\s*:|$)",
            text, re.IGNORECASE | re.DOTALL,
        )
        return m.group(1).strip() if m else text.strip()

    def _parse_action(self, text: str) -> Optional[Dict]:
        # Strategy 1
        m = re.search(r"Action\s*:\s*(\{.+?\})\s*$", text, re.IGNORECASE | re.DOTALL | re.MULTILINE)
        raw = m.group(1) if m else None
        # Strategy 2
        if raw is None:
            m = re.search(r"Action\s*:\s*```(?:json)?\s*(\{.+?\})\s*```", text, re.IGNORECASE | re.DOTALL)
            raw = m.group(1) if m else None
        # Strategy 3
        if raw is None:
            m = re.search(r"Action\s*:.*?(\{[^{}]*\})", text, re.IGNORECASE | re.DOTALL)
            raw = m.group(1) if m else None
        if raw is None:
            return None
        try:
            parsed = json.loads(raw)
            if "tool" not in parsed:
                return None
            if "args" not in parsed:
                parsed["args"] = {}
            return parsed
        except json.JSONDecodeError:
            return None

    # ──────────────────────────────────────────────
    # Tool Execution
    # ──────────────────────────────────────────────

    def _execute_tool(self, tool_name: str, args: Dict) -> str:
        tool = self.tools[tool_name]
        func: Optional[Callable] = tool.get("function")
        if func is None:
            return f"[ERROR] Tool '{tool_name}' has no callable registered."
        try:
            start = time.time()
            result = func(**args) if isinstance(args, dict) else func(args)
            elapsed = int((time.time() - start) * 1000)
            logger.log_event("TOOL_EXECUTION", {
                "version": self.VERSION,
                "tool": tool_name,
                "execution_time_ms": elapsed,
                "success": True,
            })
            return str(result)
        except TypeError as e:
            logger.log_event("TOOL_ARG_ERROR", {"tool": tool_name, "args": args, "error": str(e)})
            return f"[ERROR] Invalid arguments for '{tool_name}': {e}. Check the tool description."
        except Exception as e:
            logger.log_event("TOOL_RUNTIME_ERROR", {"tool": tool_name, "error": str(e)})
            return f"[ERROR] Tool '{tool_name}' raised: {e}"


# Factory
def create_agent_v2(llm, tools, max_steps=5, max_retries=3) -> ReActAgentV2:
    return ReActAgentV2(llm=llm, tools=tools, max_steps=max_steps, max_retries=max_retries)
