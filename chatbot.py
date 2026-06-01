import os
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
from src.core.openai_provider import OpenAIProvider
from src.core.gemini_provider import GeminiProvider
from src.core.local_provider import LocalProvider

# Load environment variables
load_dotenv()

def get_provider():
    """Factory function to get the configured LLM provider."""
    provider_name = os.getenv("DEFAULT_PROVIDER", "openai").lower()
    
    if provider_name == "openai":
        return OpenAIProvider(
            model_name=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY")
        )
    elif provider_name == "gemini":
        return GeminiProvider(
            model_name=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
            api_key=os.getenv("GEMINI_API_KEY")
        )
    elif provider_name == "local":
        return LocalProvider(
            model_path=os.getenv("LOCAL_MODEL_PATH", "models/Phi-3-mini-4k-instruct-q4.gguf")
        )
    else:
        raise ValueError(f"Unknown provider: {provider_name}")

def run_test_cases():
    try:
        provider = get_provider()
        print(f"Using provider: {provider.model_name} ({provider.__class__.__name__})")
    except Exception as e:
        print(f"Error initializing provider: {e}")
        return

    # Test cases that require multi-step reasoning or tool usage
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
    print("🤖 CHATBOT BASELINE TEST")
    print("="*50)

    for case in test_cases:
        print(f"\n[Test Case {case['id']}: {case['desc']}]")
        print(f"User: {case['prompt']}")
        print("-" * 50)
        
        try:
            # Add a basic system prompt for context
            system_prompt = "You are a helpful E-commerce Assistant. Answer the user's questions to the best of your ability."
            
            response = provider.generate(prompt=case['prompt'], system_prompt=system_prompt)
            
            print(f"Bot:\n{response['content']}")
            print(f"\n[Metrics] Latency: {response['latency_ms']}ms | Tokens: {response['usage']}")
            
        except Exception as e:
            print(f"❌ Error during generation: {e}")
        
        print("="*50)
        
    print("\nNote: Please review the outputs. Test Case 3 is expected to hallucinate or fail because this baseline chatbot does not have access to real tools.")

if __name__ == "__main__":
    run_test_cases()
