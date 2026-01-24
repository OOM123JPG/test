import os
import json
import logging
import pandas as pd
from lm_eval import evaluator
from lm_eval.tasks import TaskManager
from lm_eval.models.openai_completions import LocalCompletionsAPI
from lm_eval.utils import handle_non_serializable

# 配置日志输出，设为 INFO 级别以查看进度
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 1. LMEH 通用任务 (通常为 0-shot)
general_tasks = [
    "winogrande",
    "piqa",
    "arc_easy",
    "arc_challenge"
]

# 2. MMLU 任务 (STEM 类别，需要 fewshot=5)
mmlu_tasks = [
    "mmlu_task_astronomy",
    "mmlu_task_college_biology",
    "mmlu_task_college_chemistry",
    "mmlu_task_college_computer_science",
    "mmlu_task_college_mathematics",
    "mmlu_task_computer_security",
    "mmlu_task_conceptual_physics",
    "mmlu_task_electrical_engineering",
    "mmlu_task_high_school_biology",
    "mmlu_task_high_school_chemistry",
    "mmlu_task_high_school_computer_science",
    "mmlu_task_high_school_physics",
    "mmlu_task_high_school_statistics",
    "mmlu_task_machine_learning"
]

# 3. C-Eval 任务 (需要 fewshot=5)
ceval_tasks = [
    "ceval_task_computer_network",
    "ceval_task_college_programming",
    "ceval_task_college_physics",
    "ceval_task_college_chemistry",
    "ceval_task_discrete_mathematics",
    "ceval_task_high_school_physics",
    "ceval_task_high_school_chemistry",
    "ceval_task_high_school_biology",
    "ceval_task_middle_school_mathematics",
    "ceval_task_middle_school_biology",
    "ceval_task_middle_school_physics",
    "ceval_task_middle_school_chemistry"
]

def run_api_eval(
    model_name: str,
    base_url: str,
    tokenizer_path: str,
    output_dir: str,  # 外面传进来的基础路径
    task_names: list = None,
    batch_size: int = 1,
    num_concurrent: int = 10,
    limit: int = 1,
    num_fewshot: int = 0,
    max_gen_toks: int = 256,
    temperature: float = 0.0, 
    **kwargs
):
    """
    运行 API 评测脚本，支持路径复用和结果自动保存
    """
    if task_names is None:
        task_names = ["piqa"]

    # 1. 确保目录结构存在
    details_dir = os.path.join(output_dir, "details")
    os.makedirs(details_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "results_api.csv")

    # 2. 初始化模型
    lm_obj = LocalCompletionsAPI(
        model=model_name,
        base_url=base_url,
        tokenizer=tokenizer_path,
        tokenizer_backend="huggingface",
        batch_size=batch_size,
        num_concurrent=num_concurrent,
        max_gen_toks=max_gen_toks,
        temperature=temperature,
        tokenized_requests=False,
        max_retries=3,
        **kwargs
    )

    task_manager = TaskManager()
    all_results = []

    # 3. 循环执行任务
    for task in task_names:
        logger.info(f"\n{'='*50}\n开始评估任务: {task} (limit: {limit})\n{'='*50}")

        try:
            # 调用 evaluator.py 中的核心评估逻辑
            results = evaluator.simple_evaluate(
                model=lm_obj,
                tasks=[task],
                num_fewshot=num_fewshot,
                limit=limit,
                task_manager=task_manager,
                log_samples=True # 记录模型生成的样本详情
            )
        except Exception as e:
            logger.error(f"任务 {task} 运行失败: {e}")
            continue

        # 4. 提取指标
        task_res = results["results"].get(task, {})
        acc = task_res.get("acc,none") or task_res.get("acc")
        acc_norm = task_res.get("acc_norm,none") or task_res.get("acc_norm")

        res_row = {
            "task": task,
            "fewshot": num_fewshot,
            "acc(%)": f"{acc*100:.2f}%" if acc is not None else "N/A",
            "acc_norm(%)": f"{acc_norm*100:.2f}%" if acc_norm is not None else "N/A",
        }
        df_row = pd.DataFrame([res_row])
        print(f"\n>>> 任务 {task} 结果预览:\n{df_row}\n{'-'*30}")

        # 5. 保存结果：写入 CSV (追加模式)
        df_row.to_csv(csv_path, index=False, mode='a', header=not os.path.exists(csv_path))
        
        # 6. 保存结果：写入详细 JSON (包含 samples)
        json_output_path = os.path.join(details_dir, f"detail_{task}.json")
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=handle_non_serializable, ensure_ascii=False)
        
        logger.info(f"任务 {task} 的详细结果已保存至: {json_output_path}")
        all_results.append(res_row)

    return pd.DataFrame(all_results)

if __name__ == "__main__":
    # 统一路径配置
    BASE_OUTPUT_PATH = "/nfs-share/wx1463835/tdmoe/output/evaluation"
    
    COMMON_CONFIG = {
        "model_name": "deepseek-v3-tucker",
        "tokenizer_path": "/nfs-share/wx1463835/download/model/DeepSeek-V3-bf16",
        "base_url": "http://127.0.0.1:8888/v1/completions",
        "output_dir": BASE_OUTPUT_PATH, # 传入基础路径
        "num_concurrent": 10,
        "batch_size": 1,
        "limit": None,
        "temperature": 0.0,
    }

    # 配置参数字典：Key 是组名，Value 是该组的特定配置
    eval_groups = {
        "LMEH 通用任务": {
            "tasks": general_tasks,
            "num_fewshot": 0,
            "max_gen_toks": 16,
        },
        "MMLU 任务": {
            "tasks": mmlu_tasks,
            "num_fewshot": 5,
            "max_gen_toks": 16,
        },
        "C-Eval 任务": {
            "tasks": ceval_tasks,
            "num_fewshot": 5,
            "max_gen_toks": 256,
        },
    }

    for group_name, config in eval_groups.items():
        logger.info(f"\n>>> 运行组别: {group_name}")
        try:
            run_api_eval(
                task_names=config["tasks"], 
                num_fewshot=config["num_fewshot"], 
                max_gen_toks=config["max_gen_toks"], 
                **COMMON_CONFIG
            )
        except Exception as e:
            logger.critical(f"组别 {group_name} 崩溃: {e}")

    print(f"\n{'='*50}\n所有评估任务已结束。总表请查看: {os.path.join(BASE_OUTPUT_PATH, 'results_api.csv')}\n{'='*50}")