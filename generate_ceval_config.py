import argparse
import logging
import os
import yaml
from tqdm import tqdm

eval_logger = logging.getLogger(__name__)

# === 关键配置：请修改此处 ===
DATASET_ROOT = "/path/to/your/ceval_parquet_data" # C-Eval 离线数据根目录
TASK_PREFIX = "ceval_task"                        # 你要求的前缀
# =========================

SUBJECTS = {
    "computer_network": "计算机网络", "operating_system": "操作系统",
    "computer_architecture": "计算机组成", "college_programming": "大学编程",
    "college_physics": "大学物理", "college_chemistry": "大学化学",
    "advanced_mathematics": "高等数学", "probability_and_statistics": "概率统计",
    "discrete_mathematics": "离散数学", "electrical_engineer": "注册电气工程师",
    "metrology_engineer": "注册计量师", "high_school_mathematics": "高中数学",
    "high_school_physics": "高中物理", "high_school_chemistry": "高中化学",
    "high_school_biology": "高中生物", "middle_school_mathematics": "初中数学",
    "middle_school_biology": "初中生物", "middle_school_physics": "初中物理",
    "middle_school_chemistry": "初中化学", "veterinary_medicine": "兽医学",
    "college_economics": "大学经济学", "business_administration": "工商管理",
    "marxism": "马克思主义基本原理", "mao_zedong_thought": "毛泽东思想和中国特色社会主义理论体系概论",
    "education_science": "教育学", "teacher_qualification": "教师资格",
    "high_school_politics": "高中政治", "high_school_geography": "高中地理",
    "middle_school_politics": "初中政治", "middle_school_geography": "初中地理",
    "modern_chinese_history": "近代史纲要", "ideological_and_moral_cultivation": "思想道德修养与法律基础",
    "logic": "逻辑学", "law": "法学", "chinese_language_and_literature": "中国语言文学",
    "art_studies": "艺术学", "professional_tour_guide": "导游资格",
    "legal_professional": "法律职业资格", "high_school_chinese": "高中语文",
    "high_school_history": "高中历史", "middle_school_history": "初中历史",
    "civil_servant": "公务员", "sports_science": "体育学", "plant_protection": "植物保护",
    "basic_medicine": "基础医学", "clinical_medicine": "临床医学",
    "urban_and_rural_planner": "注册城乡规划师", "accountant": "注册会计师",
    "fire_engineer": "注册消防工程师", "environmental_impact_assessment_engineer": "环境影响评价工程师",
    "tax_accountant": "税务师", "physician": "医师资格",
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_yaml_path", default="default/_default_template_yaml")
    parser.add_argument("--save_prefix_path", default="ceval_task_configs")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    base_yaml_name = os.path.split(args.base_yaml_path)[-1]
    os.makedirs(args.save_prefix_path, exist_ok=True)

    for subject_eng, subject_zh in tqdm(SUBJECTS.items()):
        # 构造本地 Subject 文件夹路径
        subject_dir = os.path.join(DATASET_ROOT, subject_eng)
        
        description = f"以下是中国关于{subject_zh}的单项选择题，请选出其中的正确答案。\n\n"

        yaml_dict = {
            "include": f"../{args.base_yaml_path}", # 引用模板
            "task": f"{TASK_PREFIX}_{subject_eng}",
            "dataset_name": subject_eng,
            # --- 离线配置：解决 KeyError 并指向本地文件 ---
            "dataset_path": "parquet", 
            "dataset_kwargs": {
                "data_files": {
                    "dev": os.path.join(subject_dir, "dev.parquet"),
                    "val": os.path.join(subject_dir, "val.parquet"),
                    "test": os.path.join(subject_dir, "test.parquet"),
                }
            },
            "validation_split": "val",
            "fewshot_split": "dev", # 确保此处的 key 在上面的 data_files 中定义过
            # -------------------------------------------
            "description": description,
        }

        file_save_path = os.path.join(args.save_prefix_path, f"{subject_eng}.yaml")
        with open(file_save_path, "w", encoding="utf-8") as yaml_file:
            yaml.dump(yaml_dict, yaml_file, width=float("inf"), allow_unicode=True, default_style='"')

    # 生成总的 Group 配置文件
    group_yaml_dict = {
        "group": f"{TASK_PREFIX}",
        "task": [f"{TASK_PREFIX}_{task_name}" for task_name in SUBJECTS.keys()],
        "aggregate_metric_list": [
            {"metric": "acc", "aggregation": "mean", "weight_by_size": True},
        ],
        "metadata": {"version": 1.0},
    }

    with open(os.path.join(args.save_prefix_path, f"_{TASK_PREFIX}.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(group_yaml_dict
