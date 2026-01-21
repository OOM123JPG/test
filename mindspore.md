export MS_VERSION=2.7.2
pip install mindspore==${MS_VERSION} -i https://repo.mindspore.cn/pypi/simple --trusted-host repo.mindspore.cn --extra-index-url https://repo.huaweicloud.com/repository/pypi/simple/


# 自动设置 CANN 相关的环境变量
LOCAL_ASCEND=/usr/local/Ascend
source ${LOCAL_ASCEND}/ascend-toolkit/set_env.sh

# 设置日志级别（可选，2 为 Warning，方便调试）
export GLOG_v=2

python -c "import mindspore;mindspore.set_device('Ascend');mindspore.run_check()"


msrun --worker_num=2 --local_worker_num=2 --master_port=8118 --join=True python toy_tp_lowrank.py
