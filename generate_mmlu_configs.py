import os
import yaml
from tqdm import tqdm

# === 配置区 ===
# 填入你服务器上 MMLU 数据的绝对路径，例如 /home/user/data/mmlu
DATASET_ROOT = "/path/to/your/mmlu_parquet_data" 
BASE_YAML = "default/_default_template_yaml"
SAVE_DIR = "mmlu_task_configs" # 生成的 YAML 存放目录
TASK_PREFIX = "mmlu_task"     # 你要求的前缀
# ==============

# 这里的 SUBJECTS 字典保持你之前的脚本内容不变
SUBJECTS = { "abstract_algebra": "stem", "anatomy": "stem", ... } 

os.makedirs(SAVE_DIR, exist_ok=True)

for subject, category in tqdm(SUBJECTS.items()):
    # 构造该学科的本地绝对路径
    subject_path = os.path.join(DATASET_ROOT, subject)
    
    description = f"The following are multiple choice questions (with answers) about {' '.join(subject.split('_'))}.\n\n"

    yaml_dict = {
        "include": f"../{BASE_YAML}", # 引用模板
        "task": f"{TASK_PREFIX}_{subject}",
        "task_alias": subject.replace("_", " "),
        "tag": f"{TASK_PREFIX}_{category}",
        "dataset_path": "parquet", # 强制使用本地 parquet 加载器
        "dataset_kwargs": {
            "data_files": {
                # 显式映射，解决 KeyError: dev 的关键
                "dev": os.path.join(subject_path, "dev.parquet"),
                "test": os.path.join(subject_path, "test.parquet"),
                "validation": os.path.join(subject_path, "validation.parquet"),
            }
        },
        "test_split": "test",
        "fewshot_split": "dev", # 这里的 dev 会对应上面 data_files 里的 dev
        "description": description,
    }

    # 保存单个学科的 YAML
    with open(f"{SAVE_DIR}/{subject}.yaml", "w", encoding="utf-8") as f:
        yaml.dump(yaml_dict, f, allow_unicode=True, default_style='"')

# 生成总的 Group YAML (用于一次性跑所有学科)
with open(f"{SAVE_DIR}/_mmlu_all.yaml", "w", encoding="utf-8") as f:
    group_config = {
        "group": f"{TASK_PREFIX}",
        "task": [f"{TASK_PREFIX}_{cat}" for cat in set(SUBJECTS.values())]
    }
    yaml.dump(group_config, f, indent=4)
