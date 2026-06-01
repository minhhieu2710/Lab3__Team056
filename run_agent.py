import os
import json
import logging
from src.agent.agent import ReActAgent
from src.telemetry.metrics import tracker
from src.telemetry.logger import logger

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Mock Provider for offline execution
class MockProvider:
    def __init__(self, model_name="mock-model"):
        self.model_name = model_name
        self.call_count = 0
        
    def generate(self, prompt, system_prompt=None):
        self.call_count += 1
        # Test Case 1: Simple Math
        if "25 * 4 + 10" in prompt:
            if self.call_count == 1:
                return {"content": "Thought: I need to calculate this.\nAction: {\"tool\": \"calculator\", \"args\": {\"expression\": \"25 * 4 + 10\"}}", "usage": {"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80}, "latency_ms": 100, "provider": "mock"}
            else:
                self.call_count = 0
                return {"content": "Thought: I have the result.\nFinal Answer: The answer is 110.", "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}, "latency_ms": 100, "provider": "mock"}
                
        # Test Case 2: E-commerce
        elif "laptop that costs $1200" in prompt:
            if self.call_count == 1:
                return {"content": "Thought: Let's calculate the discount first.\nAction: {\"tool\": \"calculator\", \"args\": {\"expression\": \"1200 * 0.85\"}}", "usage": {"prompt_tokens": 80, "completion_tokens": 30, "total_tokens": 110}, "latency_ms": 150, "provider": "mock"}
            elif self.call_count == 2:
                return {"content": "Thought: Now add the shipping fee.\nAction: {\"tool\": \"calculator\", \"args\": {\"expression\": \"1020 + 20\"}}", "usage": {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150}, "latency_ms": 150, "provider": "mock"}
            else:
                self.call_count = 0
                return {"content": "Thought: I have the final total.\nFinal Answer: The final total is $1040.", "usage": {"prompt_tokens": 160, "completion_tokens": 20, "total_tokens": 180}, "latency_ms": 150, "provider": "mock"}
                
        # Test Case 3: Tool-dependent inquiry
        elif "Macbook Pro M3" in prompt:
            if self.call_count == 1:
                return {"content": "Thought: I need to check if the item is in stock.\nAction: {\"tool\": \"check_stock\", \"args\": {\"product\": \"Macbook Pro M3\"}}", "usage": {"prompt_tokens": 60, "completion_tokens": 30, "total_tokens": 90}, "latency_ms": 120, "provider": "mock"}
            elif self.call_count == 2:
                # Deliberate hallucination
                return {"content": "Thought: Let's calculate the price.\nAction: {\"tool\": \"hallucinated_tool\", \"args\": {\"price\": 1500}}", "usage": {"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130}, "latency_ms": 120, "provider": "mock"}
            else:
                self.call_count = 0
                return {"content": "Thought: The tool doesn't exist, I'll fallback.\nFinal Answer: Yes, the Macbook Pro M3 is in stock, but I cannot calculate the final price due to a system error.", "usage": {"prompt_tokens": 140, "completion_tokens": 35, "total_tokens": 175}, "latency_ms": 120, "provider": "mock"}
        else:
            return {"content": "Final Answer: Done.", "usage": {"total_tokens": 10}, "latency_ms": 10, "provider": "mock"}

# Define tools
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    try:
        allowed_chars = "0123456789+-*/(). "
        if any(c not in allowed_chars for c in expression):
            return "Error: Invalid characters in expression."
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

def check_stock(product: str) -> str:
    """Check stock for a product."""
    if "macbook" in product.lower():
        return "Yes, Macbook Pro M3 is in stock (5 units available)."
    return f"Sorry, {product} is out of stock."

TOOLS = [
    {
        "name": "calculator",
        "description": "Evaluate a math expression. Args: expression (string)",
        "function": calculator,
    },
    {
        "name": "check_stock",
        "description": "Check if a product is in stock. Args: product (string)",
        "function": check_stock,
    }
]

def run_test_cases():
    # Attempt to load real provider, fallback to Mock if missing dependencies
    try:
        from chatbot import get_provider
        provider = get_provider()
        print(f"Using real provider: {provider.model_name}")
    except Exception as e:
        print(f"Falling back to MockProvider due to: {e}")
        provider = MockProvider()
        
    agent = ReActAgent(llm=provider, tools=TOOLS, max_steps=7)
    
    test_cases = [
        {
            "id": 1,
            "desc": "Simple Math Q&A",
            "prompt": "What is 25 * 4 + 10?"
        },
        {
            "id": 2,
            "desc": "E-commerce multi-step reasoning",
            "prompt": "I want to buy a laptop that costs $1200. I have a 15% discount coupon. The shipping fee is $20 based on my location. Can you calculate the final total for me?"
        },
        {
            "id": 3,
            "desc": "Tool-dependent inquiry (expected to fail/hallucinate)",
            "prompt": "Check if the 'Macbook Pro M3' is in stock in the 'check_stock' system, and if so, calculate the final price with a 10% discount and $15 shipping fee."
        }
    ]

    print("\n" + "="*50)
    print("🤖 REACT AGENT v1 TEST")
    print("="*50)

    for case in test_cases:
        print(f"\n[Test Case {case['id']}: {case['desc']}]")
        print(f"User: {case['prompt']}")
        print("-" * 50)
        
        result = agent.run(case['prompt'])
        print(f"\nAgent Final Answer:\n{result}")
        print("="*50)

    print("\n[Metrics Summary]")
    summary = tracker.get_summary()
    print(json.dumps(summary, indent=2))
    
if __name__ == "__main__":
    run_test_cases()
