"""Run Gemma on a simple arithmetic prompt."""

from triton_ptx.LLMs import generate
from triton_ptx.LLMs.gemma import Gemma


def main():
    response = generate(
        Gemma(),
        [{"role": "user", "content": "What is 2+2? Also what is the capital of Switzerland, please answer 10 times with the city"}],
        max_new_tokens=400,
    )
    if not response.strip():
        raise RuntimeError("Gemma produced an empty response.")
    print(response)


if __name__ == "__main__":
    main()
