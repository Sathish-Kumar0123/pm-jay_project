
import ollama

response = ollama.chat(
    model="qwen2.5:0.5b-instruct",
    messages=[
        {
            "role": "user",
            "content": "Reply with exactly: PM-JAY LOCAL MODEL WORKING"
        }
    ],
    options={
        "temperature": 0,
        "num_predict": 30
    }
)

print(response.message.content)
