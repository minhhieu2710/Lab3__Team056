"""
Unit tests for the ReAct Agent core logic and Metrics module.
Run with: python -m pytest tests/test_react_agent.py -v
"""
import json
import pytest
import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agent.agent import ReActAgent
from src.telemetry.metrics import PerformanceTracker, PRICING


# ====================================================================
# Helpers: Mock LLM Provider
# ====================================================================

class MockLLMProvider:
    """
    A fake LLM provider that returns pre-scripted responses
    in sequence, simulating a multi-step ReAct conversation.
    """
    def __init__(self, responses: list):
        self.responses = responses
        self.call_count = 0
        self.model_name = "mock-model"

    def generate(self, prompt, system_prompt=None):
        if self.call_count >= len(self.responses):
            return {
                "content": "Final Answer: I ran out of scripted responses.",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                "latency_ms": 50,
                "provider": "mock",
            }
        response = self.responses[self.call_count]
        self.call_count += 1
        return {
            "content": response,
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            "latency_ms": 200,
            "provider": "mock",
        }

    def stream(self, prompt, system_prompt=None):
        yield "mock stream"


# Sample tools for testing
def mock_search(query: str) -> str:
    return f"Search results for '{query}': Python is a programming language."

def mock_calculator(expression: str) -> str:
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

SAMPLE_TOOLS = [
    {
        "name": "search",
        "description": "Search the web. Args: query (string)",
        "function": mock_search,
    },
    {
        "name": "calculator",
        "description": "Evaluate a math expression. Args: expression (string)",
        "function": mock_calculator,
    },
]


# ====================================================================
# Tests: ReAct Agent
# ====================================================================

class TestReActAgent:
    """Tests for the core ReAct loop logic."""

    def test_final_answer_on_first_step(self):
        """Agent returns immediately if LLM gives Final Answer on step 1."""
        llm = MockLLMProvider([
            "Thought: This is a simple greeting.\nFinal Answer: Hello! How can I help?"
        ])
        agent = ReActAgent(llm=llm, tools=SAMPLE_TOOLS)
        result = agent.run("Hi there")
        assert "Hello! How can I help?" in result

    def test_single_tool_call(self):
        """Agent calls one tool then gives final answer."""
        llm = MockLLMProvider([
            'Thought: I need to search.\nAction: {"tool": "search", "args": {"query": "Python"}}',
            "Thought: I now have all the information needed.\nFinal Answer: Python is a programming language.",
        ])
        agent = ReActAgent(llm=llm, tools=SAMPLE_TOOLS)
        result = agent.run("What is Python?")
        assert "Python is a programming language" in result
        assert llm.call_count == 2

    def test_multi_step_tool_calls(self):
        """Agent chains multiple tools before final answer."""
        llm = MockLLMProvider([
            'Thought: I need to search first.\nAction: {"tool": "search", "args": {"query": "price of item"}}',
            'Thought: Now calculate with tax.\nAction: {"tool": "calculator", "args": {"expression": "100 * 1.1"}}',
            "Thought: I now have all the information needed.\nFinal Answer: The total price with tax is 110.0",
        ])
        agent = ReActAgent(llm=llm, tools=SAMPLE_TOOLS, max_steps=5)
        result = agent.run("Calculate price with 10% tax")
        assert "110.0" in result
        assert llm.call_count == 3

    def test_max_steps_timeout(self):
        """Agent stops when max_steps is reached."""
        llm = MockLLMProvider([
            'Thought: Searching.\nAction: {"tool": "search", "args": {"query": "test"}}',
            'Thought: Searching again.\nAction: {"tool": "search", "args": {"query": "test2"}}',
            'Thought: Searching more.\nAction: {"tool": "search", "args": {"query": "test3"}}',
        ])
        agent = ReActAgent(llm=llm, tools=SAMPLE_TOOLS, max_steps=2)
        result = agent.run("Keep searching")
        assert "AGENT TIMEOUT" in result

    def test_hallucinated_tool(self):
        """Agent handles gracefully when LLM invents a non-existent tool."""
        llm = MockLLMProvider([
            'Thought: I need weather data.\nAction: {"tool": "weather_api", "args": {"city": "Hanoi"}}',
            "Thought: That tool doesn't exist, let me answer directly.\nFinal Answer: I cannot check weather without the right tool.",
        ])
        agent = ReActAgent(llm=llm, tools=SAMPLE_TOOLS)
        result = agent.run("What's the weather?")
        assert "cannot" in result.lower() or "Final Answer" in result or llm.call_count == 2

    def test_invalid_action_json_recovery(self):
        """Agent recovers when LLM outputs invalid Action format."""
        llm = MockLLMProvider([
            "Thought: Let me search.\nAction: search(query='test')",  # invalid format
            'Thought: Let me try again.\nAction: {"tool": "search", "args": {"query": "test"}}',
            "Thought: I now have all the information needed.\nFinal Answer: Found the answer.",
        ])
        agent = ReActAgent(llm=llm, tools=SAMPLE_TOOLS, max_steps=5)
        result = agent.run("Search for test")
        assert "Found the answer" in result

    def test_history_accumulates(self):
        """Each step's observation is fed back to the LLM."""
        llm = MockLLMProvider([
            'Thought: Step 1.\nAction: {"tool": "search", "args": {"query": "hello"}}',
            "Thought: I now have all the information needed.\nFinal Answer: Done.",
        ])
        agent = ReActAgent(llm=llm, tools=SAMPLE_TOOLS)
        agent.run("Test history")
        assert len(agent.history) >= 1
        assert "Observation" in agent.history[0]


# ====================================================================
# Tests: Parsing
# ====================================================================

class TestParsing:
    """Tests for the agent's parsing helpers."""

    def setup_method(self):
        self.agent = ReActAgent(
            llm=MockLLMProvider([]),
            tools=SAMPLE_TOOLS,
        )

    def test_parse_final_answer(self):
        text = "Thought: Done.\nFinal Answer: The answer is 42."
        result = self.agent._parse_final_answer(text)
        assert result == "The answer is 42."

    def test_parse_final_answer_missing(self):
        text = "Thought: Still thinking.\nAction: {\"tool\": \"search\"}"
        result = self.agent._parse_final_answer(text)
        assert result is None

    def test_parse_action_valid_json(self):
        text = 'Thought: Need to search.\nAction: {"tool": "search", "args": {"query": "test"}}'
        result = self.agent._parse_action(text)
        assert result is not None
        assert result["tool"] == "search"
        assert result["args"]["query"] == "test"

    def test_parse_action_with_markdown_fences(self):
        text = 'Thought: Need to search.\nAction: ```json\n{"tool": "search", "args": {"query": "test"}}\n```'
        result = self.agent._parse_action(text)
        assert result is not None
        assert result["tool"] == "search"

    def test_parse_action_invalid(self):
        text = "Thought: Hmm.\nAction: call the search tool please"
        result = self.agent._parse_action(text)
        assert result is None

    def test_parse_thought(self):
        text = "Thought: I should use the calculator tool.\nAction: {\"tool\": \"calculator\"}"
        result = self.agent._parse_thought(text)
        assert "calculator tool" in result


# ====================================================================
# Tests: Metrics & Cost Calculation
# ====================================================================

class TestPerformanceTracker:
    """Tests for the metrics / telemetry module."""

    def setup_method(self):
        self.tracker = PerformanceTracker()

    def test_calculate_cost_gpt4o(self):
        usage = {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500}
        cost = self.tracker.calculate_cost("gpt-4o", usage)
        # Expected: (1000/1000)*0.0025 + (500/1000)*0.01 = 0.0025 + 0.005 = 0.0075
        assert abs(cost - 0.0075) < 0.0001

    def test_calculate_cost_gemini_flash(self):
        usage = {"prompt_tokens": 2000, "completion_tokens": 1000, "total_tokens": 3000}
        cost = self.tracker.calculate_cost("gemini-1.5-flash", usage)
        # Expected: (2000/1000)*0.000075 + (1000/1000)*0.0003 = 0.00015 + 0.0003 = 0.00045
        assert abs(cost - 0.00045) < 0.00001

    def test_calculate_cost_unknown_model(self):
        """Unknown models should use the fallback pricing."""
        usage = {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500}
        cost = self.tracker.calculate_cost("totally-unknown-model", usage)
        # Fallback: (1000/1000)*0.001 + (500/1000)*0.002 = 0.001 + 0.001 = 0.002
        assert abs(cost - 0.002) < 0.0001

    def test_calculate_cost_local(self):
        """Local models should be free."""
        usage = {"prompt_tokens": 5000, "completion_tokens": 2000, "total_tokens": 7000}
        cost = self.tracker.calculate_cost("local", usage)
        assert cost == 0.0

    def test_track_request(self):
        usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        metric = self.tracker.track_request("openai", "gpt-4o", usage, 500)
        assert metric["provider"] == "openai"
        assert metric["total_tokens"] == 150
        assert metric["latency_ms"] == 500
        assert metric["cost_estimate_usd"] > 0
        assert len(self.tracker.session_metrics) == 1

    def test_token_efficiency(self):
        usage = {"prompt_tokens": 800, "completion_tokens": 200, "total_tokens": 1000}
        efficiency = PerformanceTracker._token_efficiency(usage)
        assert efficiency == 0.2  # 200/1000

    def test_token_efficiency_zero_total(self):
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        efficiency = PerformanceTracker._token_efficiency(usage)
        assert efficiency == 0.0

    def test_get_summary(self):
        usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        self.tracker.track_request("openai", "gpt-4o", usage, 200)
        self.tracker.track_request("openai", "gpt-4o", usage, 400)
        self.tracker.track_request("google", "gemini-1.5-flash", usage, 150)

        summary = self.tracker.get_summary()
        assert summary["request_count"] == 3
        assert summary["total_tokens"] == 450
        assert summary["total_cost_usd"] > 0
        assert "latency_p50_ms" in summary

    def test_get_summary_empty(self):
        summary = self.tracker.get_summary()
        assert summary["status"] == "no_data"
        assert summary["request_count"] == 0

    def test_comparison_table(self):
        usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        self.tracker.track_request("openai", "gpt-4o", usage, 300)
        self.tracker.track_request("google", "gemini-1.5-flash", usage, 150)

        table = self.tracker.get_comparison_table()
        assert len(table) == 2
        providers = {row["provider"] for row in table}
        assert "openai" in providers
        assert "google" in providers

    def test_reset(self):
        usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        self.tracker.track_request("openai", "gpt-4o", usage, 200)
        assert len(self.tracker.session_metrics) == 1

        self.tracker.reset()
        assert len(self.tracker.session_metrics) == 0

    def test_pricing_table_has_required_models(self):
        """Ensure all commonly used models have pricing entries."""
        required = ["gpt-4o", "gpt-4o-mini", "gemini-1.5-flash", "gemini-1.5-pro", "local"]
        for model in required:
            assert model in PRICING, f"Missing pricing for {model}"
            assert "prompt" in PRICING[model]
            assert "completion" in PRICING[model]


# ====================================================================
# Run
# ====================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
