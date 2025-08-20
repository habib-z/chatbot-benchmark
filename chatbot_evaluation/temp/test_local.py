import requests

url = "http://185.255.91.144:30080/v1/chat/completions"
headers = {"Content-Type": "application/json"}
data = {
    "model": "/models/abhishekchohan_gemma-3-12b-it-quantized-W4A16",
    "messages": [{"role": "user", "content": "Hello, are you alive?"}],
    "max_tokens": 50,
    "temperature": 0
}

resp = requests.post(url, headers=headers, json=data)
print(resp.json())
