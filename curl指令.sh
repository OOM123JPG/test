curl -X POST http://10.120.72.45:8888/v1/chat/completions \
   -H "Content-Type: application/json" \
   -d '{
     "model": "deepseek-v3-tucker",
     "messages": [{"role": "user", "content": "什么是人工智能？"}],
     "max_tokens": 50,
     "temperature": 0.7
   }'


curl http://127.0.0.1:8888/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v3-tucker",
    "prompt": "The goal is to test loglikelihood.",
    "temperature": 0,
    "max_tokens": 1,
    "logprobs": 1,
    "echo": true
  }'	
	
	
	   
