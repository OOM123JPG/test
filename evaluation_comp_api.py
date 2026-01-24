import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['ASCEND_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_NPU_ALLOC_CONF'] = "expandable_segments:True"
os.environ['ACL_STREAM_TIMEOUT'] = '6000'

import json
from evalscope.run import run_task
from evalscope.config import TaskConfig



work_dir_base = '/home/GZGKD001/tmp/yanhong/tdmoe_deepseek/output/evaluation_/eval_evalscope_36_layers'
os.makedirs(work_dir_base, exist_ok=True)
print("Evaluation results will be saved to:", work_dir_base)

# winogrande
# task_cfg = TaskConfig(
#     model='deepseek-v3-compressed',
#     eval_type='openai_api', 
#     api_url='http://localhost:8000/v1',
#     api_key='EMPTY',
#     datasets=['winogrande'],
#     no_timestamp=True,
#     eval_batch_size=64,
#     limit=64,
#     # generation_config={
#     #     'max_tokens': 5,
#     #     'do_sample': True,
#     #     'temperature': 0.5,
#     #     'top_p':0.9,
#     #     'repetition_penalty':1.2,
#     #     'seed': 42,
#     # },
#     work_dir=f'{work_dir_base}/winogrande',
#     use_cache=f'{work_dir_base}/winogrande'
# )
# run_task(task_cfg)
# print(f"Winogrande evaluation completed. Results saved to {work_dir_base}/winogrande")

# piqa
# task_cfg = TaskConfig(
#     model='deepseek-v3-compressed',
#     eval_type='openai_api', 
#     api_url='http://localhost:8000/v1',
#     api_key='EMPTY',
#     datasets=['piqa'],
#     no_timestamp=True,
#     eval_batch_size=32,
#     # limit=16,
#     generation_config={
#         'max_tokens': 5,
#         # 'do_sample': True,
#         # 'temperature': 0.5,
#         # 'top_p':0.9,
#         # 'repetition_penalty':1.2,
#         # 'seed': 42,
#     },
#     work_dir=f'{work_dir_base}/piqa',
#     use_cache=f'{work_dir_base}/piqa'
# )
# run_task(task_cfg)
# print(f"PIQA evaluation completed. Results saved to {work_dir_base}/piqa")


# arc-e
# task_cfg = TaskConfig(
#     model='deepseek-v3-compressed', 
#     model_id='deepseek-v3-compressed', 
#     eval_type='openai_api', 
#     api_url='http://localhost:8000/v1/chat/completions',
#     api_key='EMPTY',
#     datasets=['arc'],
#     dataset_args={
#         'arc': {
#             'subset_list': ['ARC-Easy']
#         },
#     },
#     no_timestamp=True,
#     eval_batch_size=32,
#     # limit=64,
#     work_dir=f'{work_dir_base}/arc-easy',
#     use_cache=f'{work_dir_base}/arc-easy'
# )
# run_task(task_cfg)


# arc-c
# task_cfg = TaskConfig(
#     model='deepseek-v3-compressed', 
#     model_id='deepseek-v3-compressed', 
#     eval_type='openai_api', 
#     api_url='http://localhost:8000/v1/chat/completions',
#     api_key='EMPTY',
#     datasets=['arc'],
#     dataset_args={
#         'arc': {
#             'subset_list': ['ARC-Challenge']
#         },
#     },
#     no_timestamp=True,
#     eval_batch_size=32,
#     # limit=64,
#     work_dir=f'{work_dir_base}/arc-challenge',
#     use_cache=f'{work_dir_base}/arc-challenge'
# )
# run_task(task_cfg)


# mmlu
# mmlu_subsets = [
#     'astronomy', 'college_biology', 'college_chemistry', 
#     'college_computer_science', 'college_mathematics', 'college_physics', 
#     'computer_security', 'conceptual_physics', 'electrical_engineering', 
#     'elementary_mathematics', 'high_school_biology', 'high_school_chemistry', 
#     'high_school_computer_science', 'high_school_mathematics', 'high_school_physics', 
#     'high_school_statistics', 'machine_learning', 'abstract_algebra'
# ]

mmlu_subsets=['astronomy']
print(f">>> 开始分块循环评测 MMLU，共 {len(mmlu_subsets)} 个子集...")

for subset in mmlu_subsets:
    print(f"\n{'='*40}")
    print(f">>> 正在评测子集: {subset}")
    print(f"{'='*40}")
    
    # 为每个子集创建独立的输出目录
    subset_work_dir = os.path.join(work_dir_base, 'mmlu', subset)
    
    task_cfg = TaskConfig(
        model='deepseek-v3-compressed', 
        model_id='deepseek-v3-compressed', 
        eval_type='openai_api', 
        api_url='http://localhost:8000/v1/chat/completions',
        api_key='EMPTY',
        datasets=['mmlu'],
        dataset_args={
            'mmlu': {
                'subset_list': [subset]
            },
        },
        generation_config={
            'max_tokens': 256
        },       
        no_timestamp=True,
        eval_batch_size=8,
        limit=8,
        work_dir=subset_work_dir,
        use_cache=subset_work_dir
    )
    
    try:
        run_task(task_cfg)
        print(f">>> 子集 {subset} 评测完成。")
    except Exception as e:
        print(f">>> [错误] 子集 {subset} 运行失败: {e}")
        continue

print("\n>>> 所有 MMLU 子集评测循环结束！")



# ceval
# stem_subsets = [
#     "computer_network", "operating_system", "computer_architecture",
#     "college_programming", "college_physics", "college_chemistry",
#     "advanced_mathematics", "probability_and_statistics", "discrete_mathematics",
#     "electrical_engineer", "metrology_engineer", "high_school_mathematics",
#     "high_school_physics", "high_school_chemistry", "high_school_biology",
#     "middle_school_biology", "middle_school_physics",
#     "middle_school_chemistry", "veterinary_medicine"
# ]

# # stem_subsets=["computer_network"]

# for subset in stem_subsets:
#     print(f"\n开始评测子集: {subset}")
    
#     task_cfg = TaskConfig(
#         model='deepseek-v3-compressed',
#         model_id='deepseek-v3-compressed',
#         eval_type='openai_api', 
#         api_url='http://localhost:8000/v1/chat/completions',
#         api_key='EMPTY',
#         datasets=['ceval'],
#         dataset_args={
#             'ceval': {
#                 'subset_list': [subset],
#                 'few_shot_num': 1,
#                 'filters': {'remove_until': '</think>'},                
#             },
#         },
#         # generation_config={
#         #     'max_tokens': 128,
#         #     'temperature': 0.7,  # 降低温度减少重复
#         #     'top_p': 0.9,
#         #     'top_k': 50,
#         #     'repetition_penalty': 1.2,  # 防止重复
#         #     # 'timeout':200
#         # },
#         no_timestamp=True,
#         eval_batch_size=8,
#         # limit=16,
#         work_dir=os.path.join(f"{work_dir_base}/ceval", subset), # 为每个子集存一个独立目录
#         use_cache=os.path.join(f"{work_dir_base}/ceval", subset)
#     )

#     # 执行评测
#     try:
#         run_task(task_cfg)
#     except Exception as e:
#         print(f">>> [错误] 子集 {subset} 运行失败: {e}")
#         continue
    
#     # 评测完立刻读取该子集的报告并打印 (可选)
#     report_path = os.path.join(f"{work_dir_base}/ceval", subset, 'reports/deepseek-v3/ceval.json')
#     if os.path.exists(report_path):
#         with open(report_path, 'r') as f:
#             data = json.load(f)
#             # 提取准确率
#             acc = data.get('results', [{}])[0].get('score', 'N/A')
#             print(f"子集 {subset} 评测完成！准确率: {acc}")


# humaneval
# LOCAL_IMAGE = 'swr.cn-south-1.myhuaweicloud.com/ascendhub/mindie:2.2.RC1-800I-A3-py311-openeuler24.03-lts'

# task_cfg = TaskConfig(
#     # --- 模型与 API 配置 ---
#     model='deepseek-v3-compressed',
#     model_id='deepseek-v3-compressed',
#     api_url='http://localhost:8000/v1/chat/completions',
#     api_key='EMPTY',
#     eval_type='openai_api',
    
#     # --- 数据集配置 ---
#     datasets=['humaneval'],
#     dataset_args={
#         'humaneval': {
#             'filters': {
#                 'remove_until': '</think>',
#                 'extract': r'```python\n([\s\S]*?)\n```'
#             },
#             'aggregation': 'mean_and_pass_at_k',
#             'sandbox_config': {
#                 'image': LOCAL_IMAGE,
#                 'tools_config': {
#                     'shell_executor': {},
#                     'python_executor': {}
#                 }
#             }
#         }
#     },
    
#     # --- 生成配置 ---
#     generation_config={
#         'max_tokens': 2048,
#         'temperature': 0.0,
#         'seed': 42,
#     },
    
#     # --- 沙箱全局配置 (参考你提供的示例) ---
#     use_sandbox=True,          # 启用沙箱
#     sandbox_type='docker',     # 指定沙箱类型为 Docker
#     # 如果是本地 Docker，sandbox_manager_config 可以不写，或者保持默认 {}
#     sandbox_manager_config={}, 
#     judge_worker_num=1,        # 沙箱并行进程数
    
#     # --- 其他控制 ---
#     eval_batch_size=1,
#     no_timestamp=True,
#     work_dir='/home/GZGKD001/tmp/yanhong/tdmoe/output/evalscope_36_layers/humaneval',
#     use_cache='/home/GZGKD001/tmp/yanhong/tdmoe/output/evalscope_36_layers/humaneval'
# )

# run_task(task_cfg)


# from evalscope.perf.main import run_perf_benchmark
# from evalscope.perf.arguments import Arguments

# task_cfg = Arguments(
#     parallel=[1, 10, 50, 100, 200],
#     number=[10, 20, 100, 200, 400],
#     model='deepseek-v3-compressed',
#     url='http://127.0.0.1:8000/v1/chat/completions',
#     api='openai',
#     dataset='random',
#     min_tokens=1024,
#     max_tokens=1024,
#     prefix_length=0,
#     min_prompt_length=1024,
#     max_prompt_length=1024,
#     tokenizer_path='/home/GZGKD001/tmp/yanhong/tdmoe/output/deepseek-v3-compressed',
#     extra_args={'ignore_eos': True}
# )
# results = run_perf_benchmark(task_cfg)