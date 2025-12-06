"""
Example: Quickstart with PAII Systems
"""

from paii.app import PAIISystem

if __name__ == "__main__":
    # Create system with local embeddings
    print("Initializing PAIISystem with local embeddings...")
    system = PAIISystem(embedding_provider="local")
    
    # Add some texts
    print("\nAdding texts...")
    system.add("The quick brown fox jumps over the lazy dog")
    system.add("Machine learning is a subset of artificial intelligence")
    system.add("Python is a popular programming language")
    
    # Search
    print("\nSearching for 'machine learning'...")
    results = system.search("machine learning", top_k=2)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result.score:.3f}")
        print(f"   Text: {result.text}")
    
    # Show info
    print("\n" + "="*50)
    info = system.info()
    print("System Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
