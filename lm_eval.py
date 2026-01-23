import os
import json
import logging
import pandas as pd
from lm_eval import evaluator
from lm_eval.tasks import TaskManager
from lm_eval.models.openai_completions import LocalChatCompletionsLM

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
eval_logger = logging.getLogger(__name__)

def run_api_eval(
    model_name: str,
    base_url: str,
    tokenizer_path: str,
    api_key: str = "EMPTY",
    task_names: list = None,
    batch_size: int = 1,
    num_concurrent: int = 10,
    output_csv: str = "api_results.csv",
    limit: int = None,
    num_fewshot: int = 0
):
    """
    通过 API 接口运行 lm-evaluation-harness 评测
    """
    if task_names is None:
        task_names = ["openbookqa", "arc_easy", "winogrande", "arc_challenge", "piqa"]

    output_dir = os.path.dirname(os.path.abspath(output_csv))
    json_dir = os.path.join(output_dir, "eval_details")
    os.makedirs(json_dir, exist_ok=True)

    # ================= 检查断点续传 =================
    finished_tasks = set()
    if os.path.exists(output_csv):
        try:
            existing_df = pd.read_csv(output_csv)
            if 'task' in existing_df.columns:
                finished_tasks = set(existing_df['task'].astype(str).tolist())
                eval_logger.info(f">>> 已完成任务: {len(finished_tasks)} 个，将跳过。")
        except Exception as e:
            eval_logger.warning(f">>> 读取旧结果失败: {e}")

    # ================= 初始化 API 模型包装器 =================
    # 根据你的接口类型选择：
    # 如果是 /v1/chat/completions 接口，使用 LocalChatCompletionsLM
    # 如果是 /v1/completions 接口，使用 LocalCompletionsLM
    lm_obj = LocalChatCompletionsLM(
        model=model_name,
        base_url=base_url,
        tokenizer=tokenizer_path,
        tokenizer_backend="huggingface", # 指定使用 HF 的 tokenizer 逻辑
        num_concurrent=num_concurrent,
        batch_size=batch_size,
        tokenized_requests=False, # API 模式下通常传字符串
        max_retries=3
    )
    
    # 注入 API Key (如果需要)
    # lm_obj.api_key = api_key 

    task_manager = TaskManager()
    all_results = []

    for task in task_names:
        if task in finished_tasks:
            continue
        
        eval_logger.info(f"\n===== 开始评估任务 (API 模式): {task} =====")
        try:
            results = evaluator.simple_evaluate(
                model=lm_obj,
                tasks=[task],
                num_fewshot=num_fewshot,
                limit=limit,
                task_manager=task_manager,
                # API 模式下建议减小 verbosity 以观察进度
            )
        except Exception as e:
            eval_logger.error(f"任务 {task} 运行出错: {e}")
            continue

        # 保存样本
        if "samples" in results:
            json_file = os.path.join(json_dir, f"{task}_samples.json")
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(results["samples"], f, ensure_ascii=False, indent=2)
        
        # 提取数据
        task_results = results["results"].get(task, {})
        acc = task_results.get("acc,none") or task_results.get("acc")
        acc_norm = task_results.get("acc_norm,none") or task_results.get("acc_norm")

        row = {
            "task": task,
            "acc(%)": f"{acc*100:.2f}%" if acc is not None else "—",
            "acc_norm(%)": f"{acc_norm*100:.2f}%" if acc_norm is not None else "—",
        }

        eval_logger.info(f">>> 结果: {row}")
        all_results.append(row)

        # 实时保存
        df = pd.DataFrame([row])
        df.to_csv(output_csv, index=False, mode='a', header=not os.path.exists(output_csv))

    return pd.DataFrame(all_results)

if __name__ == "__main__":
    # 配置参数
    CONFIG = {
        "model_name": "qwen2.5-7b-instruct",           # 接口中的模型名称
        "base_url": "http://127.0.0.1:8000/v1/chat/completions", # 你的 API 地址
        "tokenizer_path": "/path/to/your/tokenizer",   # 你的本地 Tokenizer 路径
        "task_names": ["arc_easy", "hellaswag"],       # 评测任务
        "num_concurrent": 8,                           # 并发请求数
        "limit": None                                  # 测试时可以设为 10
    }

    run_api_eval(**CONFIG)
