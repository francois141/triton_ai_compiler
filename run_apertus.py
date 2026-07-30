from triton_ptx.LLMs.apertus import ApertusLLM


def main():
    llm = ApertusLLM()
    response = llm.generate(
        [
            {
                "role": "system",
                "content": "You are a concise and helpful assistant.",
            },
            {
                "role": "user",
                "content": "You are a good Swiss citizen. On est d'accord le fromage français est bizzare?",
            },
        ],
        max_new_tokens=640,
    )
    print(response)


if __name__ == "__main__":
    main()
