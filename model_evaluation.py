import os
from evalscope.run import run_task
from evalscope.config import TaskConfig

# 设置工作目录
work_dir_base = '/home/GZGKD001/tmp/yanhong/tdmoe_deepseek/output/evaluation_/eval_evalscope_36_layers'
os.makedirs(work_dir_base, exist_ok=True)

def run_mmlu_evaluation():
    """运行MMLU评测"""
    
    # MMLU子集列表
    mmlu_subsets = ['astronomy', 'college_biology', 'college_chemistry', 
                   'college_computer_science', 'college_mathematics', 'college_physics', 
                   'computer_security', 'conceptual_physics', 'electrical_engineering', 
                   'elementary_mathematics', 'high_school_biology', 'high_school_chemistry', 
                   'high_school_computer_science', 'high_school_mathematics', 'high_school_physics', 
                   'high_school_statistics', 'machine_learning', 'abstract_algebra']
    
    print(f">>> 开始分块循环评测 MMLU，共 {len(mmlu_subsets)} 个子集...")
    
    for subset in mmlu_subsets:
        print(f"\n{'='*40}")
        print(f">>> 正在评测子集: {subset}")
        print(f"{'='*40}")
        
        # 为每个子集创建独立的输出目录
        subset_work_dir = os.path.join(work_dir_base, 'mmlu', subset)
        
        task_cfg = TaskConfig(
            model='deepseek-v3-tucker',  # 模型名称
            eval_type='distributed_deepseek',  # 使用我们自定义的适配器
            api_url='http://127.0.0.1:8888',  # API服务地址
            datasets=['mmlu'],
            dataset_args={
                'mmlu': {
                    'subset_list': [subset]
                },
            },
            generation_config={
                'max_tokens': 256,
                'temperature': 0.0,  # 确定性输出
            },       
            no_timestamp=True,
            eval_batch_size=8,  # 批处理大小
            limit=100,  # 每个子集测试100个样本
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

def run_ceval_evaluation():
    """运行CEval评测"""
    
    stem_subsets = [
        "computer_network", "operating_system", "computer_architecture",
        "college_programming", "college_physics", "college_chemistry",
        "advanced_mathematics", "probability_and_statistics", "discrete_mathematics",
        "electrical_engineer", "metrology_engineer", "high_school_mathematics",
        "high_school_physics", "high_school_chemistry", "high_school_biology",
        "middle_school_biology", "middle_school_physics",
        "middle_school_chemistry", "veterinary_medicine"
    ]
    
    for subset in stem_subsets:
        print(f"\n开始评测子集: {subset}")
        
        task_cfg = TaskConfig(
            model='deepseek-v3-tucker',
            eval_type='distributed_deepseek',
            api_url='http://127.0.0.1:8888',
            datasets=['ceval'],
            dataset_args={
                'ceval': {
                    'subset_list': [subset],
                    'few_shot_num': 1,
                },
            },
            generation_config={
                'max_tokens': 2048,
                'temperature': 0.0,
                'seed': 42,
            },
            no_timestamp=True,
            eval_batch_size=4,
            limit=50,  # CEval测试较少样本
            work_dir=os.path.join(f"{work_dir_base}/ceval", subset),
            use_cache=os.path.join(f"{work_dir_base}/ceval", subset)
        )

        try:
            run_task(task_cfg)
            print(f">>> 子集 {subset} 评测完成。")
        except Exception as e:
            print(f">>> [错误] 子集 {subset} 运行失败: {e}")
            continue

def run_arc_evaluation():
    """运行ARC评测"""
    
    for arc_type in ['ARC-Easy', 'ARC-Challenge']:
        print(f"\n开始评测: {arc_type}")
        
        task_cfg = TaskConfig(
            model='deepseek-v3-tucker', 
            eval_type='distributed_deepseek', 
            api_url='http://127.0.0.1:8888',
            datasets=['arc'],
            dataset_args={
                'arc': {
                    'subset_list': [arc_type]
                },
            },
            generation_config={
                'max_tokens': 256,
                'temperature': 0.0,
            },
            no_timestamp=True,
            eval_batch_size=8,
            limit=100,
            work_dir=f'{work_dir_base}/arc_{arc_type.lower().replace("-", "_")}',
            use_cache=f'{work_dir_base}/arc_{arc_type.lower().replace("-", "_")}'
        )
        
        try:
            run_task(task_cfg)
            print(f">>> {arc_type} 评测完成。")
        except Exception as e:
            print(f">>> [错误] {arc_type} 运行失败: {e}")

if __name__ == '__main__':
    # 确保API服务正在运行
    print("请确保 api_inference.py 服务正在运行在 http://127.0.0.1:8888")
    print("使用命令: python api_inference.py --model_path /path/to/model --whitening_dir /path/to/data --port 8888")
    print()
    
    # 运行不同评测
    # run_mmlu_evaluation()
    # run_ceval_evaluation() 
    run_arc_evaluation()