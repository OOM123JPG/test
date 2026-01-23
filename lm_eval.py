import os
import json
import logging
import pandas as pd
import torch
from lm_eval import evaluator
from lm_eval.tasks import TaskManager
from lm_eval.models.openai_completions import LocalCompletionsAPI

# 配置日志输出
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_api_eval(
    model_name: str,
    base_url: str,
    tokenizer_path: str,
    task_names: list = None,
    batch_size: int = 1,
    limit: int = 1,
    num_fewshot: int = 0,
    output_csv: str = "results_api.csv"
):
    """
    运行 API 评测脚本，支持 loglikelihood 任务 (如 piqa)
    """
    if task_names is None:
        task_names = ["piqa"]

    # 准备输出目录
    output_dir = os.path.dirname(os.path.abspath(output_csv)) or "."
    json_dir = os.path.join(output_dir, "eval_details")
    os.makedirs(json_dir, exist_ok=True)

    # ================= 1. 检查已完成的任务 (断点续传) =================
    finished_tasks = set()
    if os.path.exists(output_csv):
        try:
            existing_df = pd.read_csv(output_csv)
            if 'task' in existing_df.columns:
                finished_tasks = set(existing_df['task'].astype(str).tolist())
                logger.info(f">>> 检测到结果文件，已跳过任务: {finished_tasks}")
        except Exception as e:
            logger.warning(f">>> 读取现有 CSV 失败: {e}")

    # ================= 2. 初始化 API 模型包装器 =================
    # 使用 LocalCompletionsAPI 以支持 piqa 等多选任务所需的 loglikelihood
    lm_obj = LocalCompletionsAPI(
        model=model_name,
        base_url=base_url,
        tokenizer=tokenizer_path,
        tokenizer_backend="huggingface",  # 确保使用本地 HF Tokenizer
        batch_size=batch_size,
        tokenized_requests=False,         # API 模式下发送 Raw Text
        max_retries=3
    )

    task_manager = TaskManager()
    all_results = []

    # ================= 3. 循环执行任务 =================
    for task in task_names:
        if task in finished_tasks:
            continue

        logger.info(f"\n" + "="*50)
        logger.info(f"开始评估任务: {task} (limit={limit}, batch_size={batch_size})")
        logger.info("="*50)

        try:
            results = evaluator.simple_evaluate(
                model=lm_obj,
                tasks=[task],
                num_fewshot=num_fewshot,
                limit=limit,
                task_manager=task_manager,
                log_samples=True
            )
        except Exception as e:
            logger.error(f"任务 {task} 失败: {e}")
            continue

        # 保存详细样本数据
        if "samples" in results:
            sample_file = os.path.join(json_dir, f"{task}_samples.json")
            with open(sample_file, "w", encoding="utf-8") as f:
                json.dump(results["samples"], f, ensure_ascii=False, indent=2)

        # 提取指标数据
        task_res = results["results"].get(task, {})
        # piqa 常用指标是 acc 或 acc_norm
        acc = task_res.get("acc,none") or task_res.get("acc")
        acc_norm = task_res.get("acc_norm,none") or task_res.get("acc_norm")

        res_row = {
            "task": task,
            "acc(%)": f"{acc*100:.2f}%" if acc is not None else "N/A",
            "acc_norm(%)": f"{acc_norm*100:.2f}%" if acc_norm is not None else "N/A",
        }

        print(f"\n>>> [{task}] 评测完成!")
        print(f"指标结果: {res_row}")
        all_results.append(res_row)

        # 实时保存到 CSV
        df_row = pd.DataFrame([res_row])
        df_row.to_csv(output_csv, index=False, mode='a', header=not os.path.exists(output_csv))

    logger.info("\n所有评估任务已结束。")
    return pd.DataFrame(all_results)

if __name__ == "__main__":
    # 配置你的 API 和路径
    PARAMS = {
        "model_name": "your-model-name",               # 模型在 API 中的注册名
        "base_url": "http://127.0.0.1:8000/v1/completions", # 必须使用 completions 结尾
        "tokenizer_path": "/path/to/your/tokenizer",   # 本地 Tokenizer 路径
        "task_names": ["piqa"],                        # 可以改为 ["piqa", "arc_easy", "winogrande"]
        "limit": 1,                                    # 样本限制
        "batch_size": 1,                               # 批次大小
        "output_csv": "api_eval_results.csv"
    }

    run_api_eval(**PARAMS)
