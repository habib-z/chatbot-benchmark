try:
    # ✅ LangChain Community version (preferred if you're using LangChain)
    from langchain_community.llms import Replicate

    llm = Replicate(
        model="meta/llama-2-7b-chat",
        model_kwargs={"temperature": 0.7, "max_length": 500}
    )

    print("Using Replicate via langchain-community")

except ImportError:
    # ✅ Fallback: raw Replicate client
    import replicate

    output = replicate.run(
        "meta/llama-2-7b-chat",
        input={"prompt": "Hello from Replicate!", "temperature": 0.7}
    )
    print("Using raw replicate client:", "".join(output))
