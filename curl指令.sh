curl http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v3-tucker",
    "messages": [{"role": "user", "content": "1+1等于几？"}],
    "temperature": 0.0,
    "max_tokens": 50
  }'

curl http://localhost:8888/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": ["北京的特产有哪些？", "用 Python 写一个快排。"],
    "max_tokens": 128
  }'
  
curl http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v3-tucker",
    "messages": [{"role": "user", "content": "请详细解释为什么天空是蓝色的？"}],
    "temperature": 0.7,
    "repetition_penalty": 1.0,
    "max_tokens": 500
  }'


curl http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v3-tucker",
    "messages": [
      {"role": "user", "content": "你好，请问你是谁？"}
    ],
    "max_tokens": 32,
    "temperature": 0.7
  }'

  curl http://localhost:8888/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v3-tucker",
    "prompt": "中国的首都是",
    "max_tokens": 16,
    "temperature": 0.0
  }'

  curl http://localhost:8888/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v3-tucker",
    "prompt": ["人工智能的未来是", "深度学习的核心是"],
    "max_tokens": 10
  }'

curl http://localhost:8888/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v3-tucker",
    "prompt": "The capital of France is Paris",
    "logprobs": 1,
    "echo": true,
    "temperature": 0.0
  }'

curl http://localhost:8888/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v3-tucker",
    "prompt": ["The capital of France is Paris", "The capital of Japan is Tokyo"],
    "logprobs": 1,
    "max_tokens": 0,
    "echo": true
  }'
  
# curl -X POST http://127.0.0.1:8888/v1/chat/completions \
#    -H "Content-Type: application/json" \
#    -d '{
#      "model": "deepseek-v3-tucker",
#      "messages": [{"role": "user", "content": "什么是人工智能？"}],
#      "max_tokens": 50,
#      "temperature": 0.7
#    }'


# curl http://127.0.0.1:8888/v1/completions \
#   -H "Content-Type: application/json" \
#   -d '{
#     "model": "deepseek-v3-tucker",
#     "prompt": "The goal is to test loglikelihood.",
#     "temperature": 0,
#     "max_tokens": 1,
#     "logprobs": 1,
#     "echo": true
#   }'	
	
	
	   
# curl http://127.0.0.1:8888/v1/completions \
#   -H "Content-Type: application/json" \
#   -d '{
#     "model": "deepseek-v3-tucker",
#     "prompt": ["Option A is better.", "Option B is better."],
#     "temperature": 0,
#     "max_tokens": 1,
#     "logprobs": 1,
#     "echo": true
#   }'
