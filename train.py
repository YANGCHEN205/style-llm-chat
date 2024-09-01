import os
import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/root/autodl-tmp/LLaMA-Factory/src')
from src.llamafactory.train.tuner import run_exp
import yaml


def main(yaml_path_):
    with open(yaml_path_, 'r', encoding='utf-8') as f:
        param = yaml.safe_load(f)
    run_exp(param)


if __name__ == "__main__":
    #1.sft指令微调
    yaml_path = '/root/autodl-tmp/LLaMA-Factory/roleplay/lora_sft.yaml'
    # 2.奖励模型训练
    # yaml_path = '/root/autodl-tmp/LLaMA-Factory/examples/roleplay/qwen2_lora_reward.yaml'
    # 3.rlhf-ppo训练
    # yaml_path = '/root/autodl-tmp/LLaMA-Factory/examples/roleplay/qwen2_lora_sft_ppo.yaml'
	
    main(yaml_path)

# CUDA_VISIBLE_DEVICES=0 python -m debugpy --listen 5678 --wait-for-client /root/autodl-tmp/LLaMA-Factory/test.py






# # -*- coding: utf-8 -*-
# # @Time    : 2024/5/17 23:21
# # @Author  : yblir
# # @File    : lyb_merge_model.py
# # explain  :
# # =======================================================
# import os
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append('/root/autodl-tmp/LLaMA-Factory/src')
# import yaml

# from src.llamafactory.train.tuner import export_model

# if __name__ == "__main__":
#     with open('/root/autodl-tmp/LLaMA-Factory/examples/roleplay/qwen2_lora_sft_merge.yaml', 'r', encoding='utf-8') as f:
#         param = yaml.safe_load(f)

#     export_model(param)

# # CUDA_VISIBLE_DEVICES=0 python -m debugpy --listen 5678 --wait-for-client /root/autodl-tmp/LLaMA-Factory/test.py