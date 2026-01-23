import os
import json
import logging
import pandas as pd
from lm_eval import evaluator
from lm_eval.tasks import TaskManager
from lm_eval.models.openai_completions import LocalCompletionsAPI

# 配置日志输出，设为 INFO 级别以查看进度
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_api_eval(
    model_name: str,
    base_url: str,
    tokenizer_path: str,
    task_names: list = None,
    batch_size: int = 1,
    num_concurrent: int = 10,
    limit: int = 1,
    num_fewshot: int = 0,
):
    """
    运行 API 评测脚本
    """
    if task_names is None:
        task_names = ["piqa"]

    # ================= 1. 初始化 API 模型包装器 =================
    # 使用 LocalCompletionsAPI 以支持 piqa 所需的 loglikelihood 功能
    # num_concurrent 将通过 **kwargs 传递给基类 TemplateAPI
    lm_obj = LocalCompletionsAPI(
        model=model_name,
        base_url=base_url,
        tokenizer=tokenizer_path,
        tokenizer_backend="huggingface",  # 使用本地 HuggingFace tokenizer
        batch_size=batch_size,
        num_concurrent=num_concurrent,    # 并发请求数
        tokenized_requests=False,         # 传输原始字符串
        max_retries=3
    )

    task_manager = TaskManager()
    all_results = []

    # ================= 2. 循环执行任务 =================
    for task in task_names:
        logger.info(f"\n" + "="*50)
        logger.info(f"开始评估任务: {task} (并发数: {num_concurrent}, limit: {limit})")
        logger.info("="*50)

        try:
            # 运行评测
            results = evaluator.simple_evaluate(
                model=lm_obj,
                tasks=[task],
                num_fewshot=num_fewshot,
                limit=limit,
                task_manager=task_manager,
            )
        except Exception as e:
            logger.error(f"任务 {task} 运行失败: {e}")
            continue

        # 提取指标数据
        task_res = results["results"].get(task, {})
        acc = task_res.get("acc,none") or task_res.get("acc")
        acc_norm = task_res.get("acc_norm,none") or task_res.get("acc_norm")

        # 构建结果行
        res_row = {
            "task": task,
            "acc(%)": f"{acc*100:.2f}%" if acc is not None else "N/A",
            "acc_norm(%)": f"{acc_norm*100:.2f}%" if acc_norm is not None else "N/A",
        }
        
        # 转换为 DataFrame 方便查看
        df_row = pd.DataFrame([res_row])
        
        # 打印 df_row
        print("\n>>> 当前任务结果预览 (df_row):")
        print(df_row)
        print("-" * 30)

        # 实时保存到 CSV (根据要求已注释)
        # df_row.to_csv("results_api.csv", index=False, mode='a', header=not os.path.exists("results_api.csv"))
        
        all_results.append(res_row)

    logger.info("\n所有任务已执行完毕。")
    return pd.DataFrame(all_results)

if __name__ == "__main__":
    # 配置参数
    CONFIG = {
        "model_name": "deepseek-v3-tucker",
        "tokenizer_path": "/nfs-share/wx1463835/download/model/DeepSeek-V3-bf16",
        "base_url": "http://127.0.0.1:8000/v1/completions",  # 请根据你的实际 API 地址修改
        "task_names": ["piqa"], 
        "num_concurrent": 10,  # 并发数
        "limit": 1,            # 仅测试 1 条数据
        "batch_size": 1,
    }

    run_api_eval(**CONFIG)
