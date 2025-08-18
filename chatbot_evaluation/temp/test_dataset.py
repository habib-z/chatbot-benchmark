import pandas as pd
data_path = "../benchmarks/asiyeh/generated_chatbot_Q&A_gemma-3-12b_without_context.csv"
df = pd.read_csv(data_path)
df.columns=['query', 'response','category','history','retrieved_contexts','distance','reference','reference_contexts']
# assume df is your DataFrame
jsonl = df.to_json(orient="records", lines=True,force_ascii=False)

# If you want to save to a file
with open("data.jsonl", "w", encoding="utf-8") as f:
    f.write(jsonl)