"""Example: run a small local model for generation using `LocalLLM`.

Requires: `transformers` installed and a small model (default: distilgpt2).
"""
from paii.llm import LocalLLM


def main():
    model = LocalLLM(model_name="distilgpt2")
    prompt = "Write a short summary about retrieval-augmented generation:\n"
    out = model.generate(prompt, max_tokens=80, temperature=0.7)
    print("--- OUTPUT ---")
    print(out)


if __name__ == "__main__":
    main()
