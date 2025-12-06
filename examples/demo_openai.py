"""
Example: Using PAII with OpenAI embeddings

To run this example:
1. Set your OpenAI API key: export OPENAI_API_KEY="sk-..."
2. Run: python examples/demo_openai.py

This will:
- Create a PAIISystem using OpenAI embeddings
- Add some texts to the index
- Search and display results with scores
"""

import os
from paii.app import PAIISystem

if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set")
        print("   To use this example, first run:")
        print("   export OPENAI_API_KEY='your-key-here'")
        print("\n   Or pass via CLI: --api-key <your-key>")
        exit(1)
    
    # Create system with OpenAI embeddings
    print("Initializing PAIISystem with OpenAI embeddings...")
    system = PAIISystem(
        embedding_provider="openai",
        api_key=api_key,
        model="text-embedding-3-small"
    )
    
    # Add some texts
    print("\nAdding texts...")
    system.add("The quick brown fox jumps over the lazy dog")
    system.add("Artificial intelligence is transforming industries worldwide")
    system.add("Python is widely used in data science and machine learning")
    
    # Search
    print("\nSearching for 'AI and technology'...")
    results = system.search("AI and technology", top_k=2)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result.score:.3f}")
        print(f"   Text: {result.text}")
    
    # Show info
    print("\n" + "="*50)
    info = system.info()
    print("System Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
