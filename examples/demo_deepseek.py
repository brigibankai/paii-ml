"""Example: call Deepseek API via `DeepseekLLM`.

Set the `DEEPSEEK_API_KEY` environment variable before running.
"""
import os
from paii.llm import DeepseekLLM


def main():
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("Set DEEPSEEK_API_KEY in the environment to run this demo.")

    model = DeepseekLLM(api_key=api_key)
    prompt = "Explain briefly how FAISS indexing works."
    out = model.generate(prompt, max_tokens=120, temperature=0.1)
    print(out)


if __name__ == "__main__":
    main()
