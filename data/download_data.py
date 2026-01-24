from modelscope.msdatasets import MsDataset
import os

# 指定保存路径
save_path = '/data/yuebin/tdmoe/test/data'
os.makedirs(save_path, exist_ok=True)

# 下载数据集到指定目录
ds = MsDataset.load(
    'AI-ModelScope/winogrande_val',
    subset_name='default',
    split='validation',
    cache_dir=save_path
)

print(f"数据集已下载到: {save_path}")

# 删除残留的 .lock 文件
for root, dirs, files in os.walk(save_path):
    for file in files:
        if file.endswith('.lock'):
            lock_path = os.path.join(root, file)
            try:
                os.remove(lock_path)
                print(f"已删除 lock 文件: {lock_path}")
            except Exception as e:
                print(f"无法删除 {lock_path}: {e}")

print("lock 文件处理完成")
