import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN not found in .env file."
    )


client = InferenceClient(
    api_key=HF_TOKEN
)


response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "user",
            "content": "Explain machine learning in one sentence."
        }
    ],
    max_tokens=100,
)

print("\nResponse:")
print(response.choices[0].message.content)