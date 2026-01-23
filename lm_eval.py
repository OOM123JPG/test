import lm_eval  
  
# API 模型评估  
results = lm_eval.simple_evaluate(  
    model="openai-completions",  
    model_args="model=gpt-3.5-turbo-instruct,api_key=YOUR_KEY",  
    tasks=["hellaswag", "arc_easy"],  
    num_fewshot=5,  
    batch_size=1  
)  
  
print(results["results"])
