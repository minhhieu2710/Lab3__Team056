import os
import re
import json
import time
from typing import List, Dict, Any, Optional, Callable
from src.core.llm_provider import LLMProvider
from src.telemetry.logger import logger
from src.telemetry.metrics import tracker


class ReActAgent:
    """
    A ReAct-style Agent that follows the Thought-Action-Observation loop.

    The ReAct paradigm (Yao et al., 2022) interleaves reasoning traces
    ("Thought") with task-specific actions ("Action"), then feeds
    environment feedback ("Observation") back into the prompt so the LLM
    can refine its plan step-by-step until it produces a "Final Answer".

    Attributes:
        llm:        An LLMProvider instance (OpenAI, Gemini, or Local).
        tools:      A list of tool dicts, each with 'name', 'description',
                    and 'function' (a callable).
        max_steps:  Safety cap to prevent infinite loops.
        history:    Accumulated Thought/Action/Observation text fed back
                    to the LLM on every iteration.
    """

    def __init__(
        self,
        llm: LLMProvider,
        tools: List[Dict[str, Any]],
        max_steps: int = 5,
    ):
        self.llm = llm
        self.tools = {t["name"]: t for t in tools}  # index by name for O(1) lookup
        self.tools_list = tools                       # keep original list for prompt
        self.max_steps = max_steps
        self.history: List[str] = []

    # ------------------------------------------------------------------
    # System Prompt
    # ------------------------------------------------------------------

    def get_system_prompt(self) -> str:
        """
        Build the system prompt that instructs the LLM to follow the
        ReAct format and lists every available tool with its description.

        The prompt enforces:
        - Structured output: Thought / Action / Final Answer
        - JSON action format so we can parse reliably
        - A clear stop condition ("Final Answer:")
        """
        tool_descriptions = "\n".join(
            [
                f'  - {t["name"]}: {t["description"]}'
                for t in self.tools_list
            ]
        )

        return (
            "You are a helpful AI assistant that solves tasks step by step.\n"
            "You have access to the following tools:\n"
            f"{tool_descriptions}\n\n"
            "You MUST follow this exact format for EVERY step:\n\n"
            "Thought: <your reasoning about what to do next>\n"
            'Action: {"tool": "<tool_name>", "args": {"<arg1>": "<value1>", ...}}\n'
            "Observation: <result will be inserted by the system>\n\n"
            "Rules:\n"
            "1. Always start with a Thought.\n"
            "2. After each Thought, output EXACTLY ONE Action line as valid JSON.\n"
            "3. Do NOT invent tool names that are not in the list above.\n"
            "4. Do NOT output an Observation yourself — the system will provide it.\n"
            "5. When you have enough information to answer, respond with:\n"
            "   Thought: I now have all the information needed.\n"
            "   Final Answer: <your complete answer to the user>\n"
            "6. Only output raw JSON in the Action line — no markdown fences.\n"
        )

    # ------------------------------------------------------------------
    # Core ReAct Loop
    # ------------------------------------------------------------------

    def run(self, user_input: str) -> str:
        """
        Execute the full ReAct loop for a given user query.

        Flow:
            1. Send (system_prompt + history + user_input) to the LLM.
            2. Parse the response for "Final Answer" → return immediately.
            3. Otherwise parse the "Action" JSON → execute the tool.
            4. Append Thought + Action + Observation to history.
            5. Repeat until Final Answer or max_steps exhausted.

        Returns:
            The agent's final answer string, or a timeout message.
        """
        logger.log_event(
            "AGENT_START",
            {
                "input": user_input,
                "model": self.llm.model_name,
                "max_steps": self.max_steps,
            },
        )

        self.history = []
        steps = 0
        total_tokens_used = 0
        total_cost = 0.0

        while steps < self.max_steps:
            steps += 1

            # ---------- 1. Build the full prompt ----------
            prompt = self._build_prompt(user_input)

            # ---------- 2. Call the LLM ----------
            try:
                result = self.llm.generate(
                    prompt, system_prompt=self.get_system_prompt()
                )
            except Exception as e:
                logger.log_event(
                    "LLM_ERROR",
                    {"step": steps, "error": str(e)},
                )
                self.history.append(
                    f"Observation: [ERROR] LLM call failed: {e}"
                )
                continue

            content: str = result.get("content", "")
            usage: dict = result.get("usage", {})
            latency_ms: int = result.get("latency_ms", 0)
            provider: str = result.get("provider", "unknown")

            # Track telemetry for this request
            tracker.track_request(provider, self.llm.model_name, usage, latency_ms)
            total_tokens_used += usage.get("total_tokens", 0)
            total_cost += tracker.calculate_cost(self.llm.model_name, usage)

            logger.log_event(
                "AGENT_STEP",
                {
                    "step": steps,
                    "llm_response": content[:500],  # truncate for log readability
                    "latency_ms": latency_ms,
                    "tokens": usage,
                },
            )

            # ---------- 3. Check for Final Answer ----------
            final_answer = self._parse_final_answer(content)
            if final_answer is not None:
                logger.log_event(
                    "AGENT_END",
                    {
                        "steps": steps,
                        "status": "final_answer",
                        "total_tokens": total_tokens_used,
                        "total_cost_usd": round(total_cost, 6),
                    },
                )
                return final_answer

            # ---------- 4. Parse Thought + Action ----------
            thought = self._parse_thought(content)
            action_json = self._parse_action(content)

            if action_json is None:
                # LLM didn't output a valid Action — log and retry
                logger.log_event(
                    "PARSE_ERROR",
                    {
                        "step": steps,
                        "error": "Could not parse Action JSON from LLM output",
                        "raw_output": content[:300],
                    },
                )
                self.history.append(content.strip())
                self.history.append(
                    "Observation: [SYSTEM] Your previous response did not contain "
                    "a valid Action JSON. Please follow the required format exactly."
                )
                continue

            tool_name = action_json.get("tool", "")
            tool_args = action_json.get("args", {})

            # ---------- 5. Execute the Tool ----------
            observation = self._execute_tool(tool_name, tool_args)

            logger.log_event(
                "TOOL_CALL",
                {
                    "step": steps,
                    "tool": tool_name,
                    "args": tool_args,
                    "observation": str(observation)[:300],
                },
            )

            # ---------- 6. Append to history ----------
            history_block = f"Thought: {thought}\n"
            history_block += f'Action: {json.dumps({"tool": tool_name, "args": tool_args})}\n'
            history_block += f"Observation: {observation}"
            self.history.append(history_block)

        # Exhausted max_steps — force a graceful exit
        logger.log_event(
            "AGENT_END",
            {
                "steps": steps,
                "status": "max_steps_exceeded",
                "total_tokens": total_tokens_used,
                "total_cost_usd": round(total_cost, 6),
            },
        )
        return (
            f"[AGENT TIMEOUT] Reached the maximum of {self.max_steps} steps "
            f"without a final answer. Last context:\n"
            + "\n".join(self.history[-2:])
        )

    # ------------------------------------------------------------------
    # Prompt Builder
    # ------------------------------------------------------------------

    def _build_prompt(self, user_input: str) -> str:
        """
        Assemble the user-facing prompt by combining the original query
        with the accumulated Thought/Action/Observation history.
        """
        parts = [f"User Query: {user_input}"]
        if self.history:
            parts.append("\n--- Previous Steps ---")
            parts.extend(self.history)
            parts.append("--- End Previous Steps ---\n")
            parts.append("Continue reasoning from where you left off.")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Parsing Helpers
    # ------------------------------------------------------------------

    def _parse_final_answer(self, text: str) -> Optional[str]:
        """
        Extract the final answer if the LLM produced one.
        Matches 'Final Answer:' (case-insensitive) followed by the answer text.
        """
        match = re.search(
            r"Final\s*Answer\s*:\s*(.+)", text, re.IGNORECASE | re.DOTALL
        )
        if match:
            return match.group(1).strip()
        return None

    def _parse_thought(self, text: str) -> str:
        """
        Extract the Thought block from the LLM output.
        If parsing fails, return the entire output as the thought.
        """
        match = re.search(
            r"Thought\s*:\s*(.+?)(?=Action\s*:|Final\s*Answer\s*:|$)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            return match.group(1).strip()
        return text.strip()

    def _parse_action(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract and parse the Action JSON from the LLM output.

        Handles common LLM quirks:
        - Markdown code fences (```json ... ```)
        - Extra whitespace or newlines inside the JSON
        - Action key with or without colon spacing
        """
        # Strategy 1: Look for Action: { ... }
        match = re.search(
            r"Action\s*:\s*(\{.+?\})\s*$", text, re.IGNORECASE | re.DOTALL | re.MULTILINE
        )
        raw_json = match.group(1) if match else None

        # Strategy 2: Look for JSON inside markdown fences
        if raw_json is None:
            match = re.search(
                r"Action\s*:\s*```(?:json)?\s*(\{.+?\})\s*```",
                text,
                re.IGNORECASE | re.DOTALL,
            )
            raw_json = match.group(1) if match else None

        # Strategy 3: Grab the first JSON-like object after "Action"
        if raw_json is None:
            match = re.search(
                r"Action\s*:.*?(\{[^{}]*\})", text, re.IGNORECASE | re.DOTALL
            )
            raw_json = match.group(1) if match else None

        if raw_json is None:
            return None

        try:
            parsed = json.loads(raw_json)
            # Validate expected keys
            if "tool" not in parsed:
                return None
            if "args" not in parsed:
                parsed["args"] = {}
            return parsed
        except json.JSONDecodeError:
            return None

    # ------------------------------------------------------------------
    # Tool Execution
    # ------------------------------------------------------------------

    def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """
        Look up a tool by name and invoke its callable with the parsed args.

        The tool dict is expected to have:
            {
                "name": "tool_name",
                "description": "What the tool does",
                "function": callable  # the actual Python function
            }

        Returns the string result of the tool, or an error message.
        """
        if tool_name not in self.tools:
            logger.log_event(
                "HALLUCINATION_ERROR",
                {"tool_requested": tool_name, "available": list(self.tools.keys())},
            )
            return (
                f"[ERROR] Tool '{tool_name}' does not exist. "
                f"Available tools: {list(self.tools.keys())}"
            )

        tool = self.tools[tool_name]
        func: Optional[Callable] = tool.get("function")

        if func is None:
            return f"[ERROR] Tool '{tool_name}' has no callable function registered."

        try:
            start = time.time()
            result = func(**args) if isinstance(args, dict) else func(args)
            elapsed_ms = int((time.time() - start) * 1000)

            logger.log_event(
                "TOOL_EXECUTION",
                {
                    "tool": tool_name,
                    "execution_time_ms": elapsed_ms,
                    "success": True,
                },
            )
            return str(result)

        except TypeError as e:
            logger.log_event(
                "TOOL_ARG_ERROR",
                {"tool": tool_name, "args": args, "error": str(e)},
            )
            return (
                f"[ERROR] Invalid arguments for tool '{tool_name}': {e}. "
                f"Please check the tool description for the correct format."
            )
        except Exception as e:
            logger.log_event(
                "TOOL_RUNTIME_ERROR",
                {"tool": tool_name, "args": args, "error": str(e)},
            )
            return f"[ERROR] Tool '{tool_name}' raised an exception: {e}"


# ------------------------------------------------------------------
# Convenience factory (used by main.py or notebooks)
# ------------------------------------------------------------------

def create_agent(
    llm: LLMProvider,
    tools: List[Dict[str, Any]],
    max_steps: int = 5,
) -> ReActAgent:
    """Create and return a configured ReActAgent instance."""
    return ReActAgent(llm=llm, tools=tools, max_steps=max_steps)
