curl http://localhost:11434/api/chat -d '{
  "model": "deepseek-r1:8b-llama-distill-q8_0",
  "messages": [
    { "role": "user", "content": "Create a short story about a robot discovering nature for the first time. Generate 1000 words" }
  ]
}'
