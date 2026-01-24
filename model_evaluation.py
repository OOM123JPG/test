import os
import json
from evalscope.run import run_task
from evalscope.config import TaskConfig

# 重要：导入自定义适配器，确保它被注册
from model_adapter import DistributedDeepSeekAdapter

# 设置工作目录
work_dir_base = '/nfs-share/wx1463835/tdmode/output/evaluation/evalscope'
os.makedirs(work_dir_base, exist_ok=True)
print("Evaluation results will be saved to:", work_dir_base)

def run_winogrande_evaluation():
    """运行Winogrande评测"""
    print("\n" + "="*50)
    print(">>> 开始评测 Winogrande")
    print("="*50)
    
    task_cfg = TaskConfig(
        model='deepseek-v3-tucker',  # 模型名称
        eval_type='distributed_deepseek',  # 使用自定义适配器
        api_url='http://127.0.0.1:8888',   # 传递给适配器的参数
        datasets=['winogrande'],
        generation_config={
            'max_tokens': 5,
            'temperature': 0.0,
        },
        no_timestamp=True,
        eval_batch_size=16,
        limit=100,
        work_dir=f'{work_dir_base}/winogrande',
        use_cache=f'{work_dir_base}/winogrande'
    )
    
    try:
        run_task(task_cfg)
        print("✓ Winogrande evaluation completed successfully")
        
        # 读取结果
        report_path = os.path.join(work_dir_base, 'winogrande', 'reports', 'deepseek-v3-tucker', 'winogrande.json')
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                data = json.load(f)
                acc = data.get('results', [{}])[0].get('score', 'N/A')
                print(f"Winogrande 准确率: {acc}")
                
    except Exception as e:
        print(f"✗ Winogrande evaluation failed: {e}")

def run_piqa_evaluation():
    """运行PIQA评测"""
    print("\n" + "="*50)
    print(">>> 开始评测 PIQA")
    print("="*50)
    
    task_cfg = TaskConfig(
        model='deepseek-v3-tucker',
        eval_type='distributed_deepseek',
        api_url='http://127.0.0.1:8888',
        datasets=['piqa'],
        generation_config={
            'max_tokens': 5,
            'temperature': 0.0,
        },
        no_timestamp=True,
        eval_batch_size=16,
        limit=100,
        work_dir=f'{work_dir_base}/piqa',
        use_cache=f'{work_dir_base}/piqa'
    )
    
    try:
        run_task(task_cfg)
        print("✓ PIQA evaluation completed successfully")
        
        # 读取结果
        report_path = os.path.join(work_dir_base, 'piqa', 'reports', 'deepseek-v3-tucker', 'piqa.json')
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                data = json.load(f)
                acc = data.get('results', [{}])[0].get('score', 'N/A')
                print(f"PIQA 准确率: {acc}")
                
    except Exception as e:
        print(f"✗ PIQA evaluation failed: {e}")

def run_arc_evaluation():
    """运行ARC评测（Easy和Challenge）"""
    print("\n" + "="*50)
    print(">>> 开始评测 ARC")
    print("="*50)
    
    for arc_type in ['ARC-Easy', 'ARC-Challenge']:
        print(f"\n>>> 评测 {arc_type}")
        
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
            print(f"✓ {arc_type} evaluation completed successfully")
            
            # 读取结果
            report_path = os.path.join(work_dir_base, f'arc_{arc_type.lower().replace("-", "_")}', 
                                     'reports', 'deepseek-v3-tucker', 'arc.json')
            if os.path.exists(report_path):
                with open(report_path, 'r') as f:
                    data = json.load(f)
                    acc = data.get('results', [{}])[0].get('score', 'N/A')
                    print(f"{arc_type} 准确率: {acc}")
                    
        except Exception as e:
            print(f"✗ {arc_type} evaluation failed: {e}")

def run_mmlu_evaluation():
    """运行MMLU评测"""
    print("\n" + "="*50)
    print(">>> 开始评测 MMLU")
    print("="*50)
    
    # MMLU子集列表
    mmlu_subsets = [
        'astronomy', 'college_biology', 'college_chemistry', 
        'college_computer_science', 'college_mathematics', 'college_physics', 
        'computer_security', 'high_school_mathematics', 'high_school_physics'
    ]
    
    print(f">>> 将评测以下 {len(mmlu_subsets)} 个 MMLU 子集: {', '.join(mmlu_subsets)}")
    
    for subset in mmlu_subsets:
        print(f"\n>>> 评测子集: {subset}")
        
        subset_work_dir = os.path.join(work_dir_base, 'mmlu', subset)
        
        task_cfg = TaskConfig(
            model='deepseek-v3-tucker', 
            eval_type='distributed_deepseek',
            api_url='http://127.0.0.1:8888',
            datasets=['mmlu'],
            dataset_args={
                'mmlu': {
                    'subset_list': [subset]
                },
            },
            generation_config={
                'max_tokens': 256,
                'temperature': 0.0,
            },       
            no_timestamp=True,
            eval_batch_size=4,
            limit=50,
            work_dir=subset_work_dir,
            use_cache=subset_work_dir
        )
        
        try:
            run_task(task_cfg)
            print(f"✓ 子集 {subset} 评测完成")
            
            # 读取结果
            report_path = os.path.join(subset_work_dir, 'reports', 'deepseek-v3-tucker', 'mmlu.json')
            if os.path.exists(report_path):
                with open(report_path, 'r') as f:
                    data = json.load(f)
                    acc = data.get('results', [{}])[0].get('score', 'N/A')
                    print(f"{subset} 准确率: {acc}")
                    
        except Exception as e:
            print(f"✗ 子集 {subset} 运行失败: {e}")
            continue
    
    print("\n>>> 所有 MMLU 子集评测循环结束！")

def run_ceval_evaluation():
    """运行CEval评测"""
    print("\n" + "="*50)
    print(">>> 开始评测 CEval")
    print("="*50)
    
    stem_subsets = [
        "computer_network", "operating_system", "college_programming", 
        "college_physics", "high_school_mathematics", "high_school_physics"
    ]
    
    print(f">>> 将评测以下 {len(stem_subsets)} 个 CEval 子集: {', '.join(stem_subsets)}")
    
    for subset in stem_subsets:
        print(f"\n>>> 评测子集: {subset}")
        
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
            limit=50,
            work_dir=os.path.join(f"{work_dir_base}/ceval", subset),
            use_cache=os.path.join(f"{work_dir_base}/ceval", subset)
        )

        try:
            run_task(task_cfg)
            print(f"✓ 子集 {subset} 评测完成")
            
            # 读取结果
            report_path = os.path.join(work_dir_base, 'ceval', subset, 'reports', 'deepseek-v3-tucker', 'ceval.json')
            if os.path.exists(report_path):
                with open(report_path, 'r') as f:
                    data = json.load(f)
                    acc = data.get('results', [{}])[0].get('score', 'N/A')
                    print(f"{subset} 准确率: {acc}")
                    
        except Exception as e:
            print(f"✗ 子集 {subset} 运行失败: {e}")
            continue

def run_all_evaluations():
    """运行所有评测"""
    print("开始运行所有评测...")
    print("请确保以下条件满足：")
    print("1. api_inference.py 服务正在运行在 http://127.0.0.1:8888")
    print("2. model_adapter.py 已导入并注册了 distributed_deepseek 适配器")
    print("="*80)
    
    try:
        # 常识推理评测
        run_winogrande_evaluation()
        run_piqa_evaluation()
        
        # 科学知识评测
        run_arc_evaluation()
        
        # 学术能力评测
        run_mmlu_evaluation()
        run_ceval_evaluation()
        
        print("\n" + "="*80)
        print("✓ 所有评测完成！")
        print(f"结果保存在: {work_dir_base}")
        
    except Exception as e:
        print(f"\n✗ 评测过程中出现错误: {e}")

if __name__ == '__main__':
    # 可以选择运行单个评测或全部评测
    # run_winogrande_evaluation()
    # run_piqa_evaluation()
    # run_arc_evaluation()
    # run_mmlu_evaluation()
    # run_ceval_evaluation()
    
    # 或者运行所有评测
    run_all_evaluations()