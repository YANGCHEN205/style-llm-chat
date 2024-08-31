import yaml
import json
from loguru import logger
import time
import sys
from src.llamafactory.chat import ChatModel
import json
import tqdm
import random

with open('/root/autodl-tmp/LLaMA-Factory/roleplay/chat_model.yaml', 'r', encoding='utf-8') as f:
    param = yaml.safe_load(f)

chat_model = ChatModel(param)


# 假设这是转换到古文的函数
def convert_to_classical_chinese(text):
    # 这里应该是将现代文本转换为古文的逻辑
    # 例如，你可以用替换规则或者调用某个外部的API进行转换
    prompt = f"把以下文字改写成三国演义原著的古文行文风格,只需要最终的结果,不需要额外的添加内容：‘{text}’ "
    messages = [
        {"role": "user", "content": prompt}
    ]
    res = chat_model.chat(messages)
    return res[0].response_text  # 暂时返回原文，需要你自己实现转换逻辑

# 读取JSON数据
with open('/root/autodl-tmp/LLaMA-Factory/data/alpaca_gpt4_data_zh.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 随机选择50条数据不重复采样
selected_data = random.sample(data, 3000)

# 处理每一个条目
for item in tqdm.tqdm(selected_data):
    # 转换output字段
    item['output'] = convert_to_classical_chinese(item['output'])

# 保存新的JSON数据到另一个文件
with open('/root/autodl-tmp/LLaMA-Factory/data/alpaca_gpt4_data_zh_modified1.json', 'w', encoding='utf-8') as f:
    json.dump(selected_data, f, ensure_ascii=False, indent=4)

print("转换完成，结果已保存。")
