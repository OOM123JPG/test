<img width="1032" height="779" alt="image" src="https://github.com/user-attachments/assets/dbe6eb53-059e-4441-8c69-c4d0b2a9ac54" />
<img width="1110" height="771" alt="image" src="https://github.com/user-attachments/assets/d8bd550e-0a4e-4408-87a5-a0ac450991f3" />
<img width="1136" height="1005" alt="image" src="https://github.com/user-attachments/assets/2fc6c9b2-bdb5-4383-84fa-b05e4717faa2" />


https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/models/deepseek-v3/README.md


docker run -it -d \
    --name deepseek_tucker \
    --net=host \
    --shm-size=128g \
    --privileged \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /home/models:/home/models \
    -v /home/your_code:/home/your_code \
    --device=/dev/davinci0 \
    --device=/dev/davinci1 \
    --device=/dev/davinci2 \
    --device=/dev/davinci3 \
    --device=/dev/davinci4 \
    --device=/dev/davinci5 \
    --device=/dev/davinci6 \
    --device=/dev/davinci7 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    ascend-mindie:latest /bin/bash
