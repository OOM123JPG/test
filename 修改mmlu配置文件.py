DATASET_ROOT = "/path/to/your/local/mmlu_parquet_root" 

    for subject, category in tqdm(SUBJECTS.items()):
        # ... 保持之前的 category 和 description 逻辑不变 ...

        # 动态构建该学科的本地 parquet 路径
        subject_dir = os.path.join(DATASET_ROOT, subject)
        
        yaml_dict = {
            "include": base_yaml_name,
            "tag": f"mmlu_{args.task_prefix}_{category}" if args.task_prefix != "" else f"mmlu_{category}",
            "task": f"mmlu_{args.task_prefix}_{subject}" if args.task_prefix != "" else f"mmlu_{subject}",
            "task_alias": subject.replace("_", " "),
            # --- 关键修改：强制使用本地离线加载配置 ---
            "dataset_path": "parquet", 
            "dataset_kwargs": {
                "data_files": {
                    "dev": os.path.join(subject_dir, "dev.parquet"),
                    "test": os.path.join(subject_dir, "test.parquet"),
                    "validation": os.path.join(subject_dir, "validation.parquet"),
                }
            },
            "test_split": "test",
            "fewshot_split": "dev",
            # ---------------------------------------
            "description": description,
        }
