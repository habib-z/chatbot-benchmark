from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(
    model="/models/abhishekchohan_gemma-3-12b-it-quantized-W4A16",
    openai_api_key="EMPTY",   # required but ignored by vLLM
    openai_api_base="http://185.255.91.144:30080/v1",  # ← use .144 not .145
    temperature=0.0,
    max_tokens=512,
)

resp = llm.invoke([HumanMessage(content="سلام، کار می‌کنی؟")])
print(resp.content)
