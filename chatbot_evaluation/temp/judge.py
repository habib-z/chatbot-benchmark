
from langchain_community.llms import Replicate
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
import os
def get_judge(model="openai/gpt-4o-mini"):
    os.environ["REPLICATE_API_TOKEN"] = ""
    llm = Replicate(
        model=model,
        model_kwargs={"temperature": 0.0, "max_length": 4096, "top_p": 1},
    )
    evaluator_llm = LangchainLLMWrapper(llm)
    return evaluator_llm

def get_local_judge(model="/models/abhishekchohan_gemma-3-12b-it-quantized-W4A16"):
    # Point to your vLLM server
    llm = ChatOpenAI(
        model=model,
        openai_api_key="EMPTY",  # vLLM ignores key, but LangChain requires it
        openai_api_base="http://185.255.91.145:30080/v1",
        temperature=0.0,
        max_tokens=4096,
    )
    evaluator_llm = LangchainLLMWrapper(llm)
    return evaluator_llm