import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import fire
import gradio as gr
from gradio import utils
import torch
from peft import PeftModel
from transformers import GenerationConfig, AutoTokenizer, AutoModel
import yaml
import json
from loguru import logger
import time
import sys
from src.llamafactory.chat import ChatModel


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass


class Chat_Model(object):
    def __init__(self, scene=None, character=None, config_path="/root/autodl-tmp/LLaMA-Factory/roleplay/single_chat_model.yaml"):
        self.device = device
        self.scene = scene
        self.config_path = config_path
        self.character = character
        self.model = self.prepare_model(self.config_path)

    def prepare_model(self, config_path):
        # load model
        with open(config_path, 'r', encoding='utf-8') as f:
            chat_model_param = yaml.safe_load(f)
        chat_model = ChatModel(chat_model_param)
        return chat_model

    # def get_headers(self):
    #     if self.scene and self.character:
    #         headers = [
    #             {"role": "user", "content": f"请扮演{self.scene}中的{self.character}。使用现代文风普通话，用{self.character}：...格式进行回复,"},
    #         ]
    #     else:
    #         headers = []
    #     return headers

    def generate(self, messages, temperature=0.9, top_p=0.75, top_k=40, num_beams=1, do_sample=True, max_new_tokens=128):
        conversation = messages
        chat_model_res = self.model.chat(
            messages=conversation,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            do_sample=do_sample,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
        return chat_model_res


def main(
    server_name: str = "0.0.0.0",
    share_gradio: bool = False,
    chat_model_config_path="/root/autodl-tmp/LLaMA-Factory/roleplay/single_chat_model.yaml"):

    chat_model = Chat_Model(scene=None, character=None, config_path=chat_model_config_path)
    
    def evaluate(
        input_text="",
        do_sample=False,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
        history=[],
        max_history=0,
        **kwargs,
    ):

        context = input_text
        if max_history > 0 and history:
            tem_history = sum(history, [])
            context = "".join(tem_history[-max_history:]) + input_text
        chat_model_prompt = context


        messages = [{"role": "user", "content": chat_model_prompt}]
        chat_model_res = chat_model.generate(
            messages,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
        )
        output = chat_model_res[0].response_text
        # output = 'werwerwer'
        history.append([input_text, output])
        return "", history

    with gr.Blocks() as demo:
        title = "💬Style-LLM-Chat-Single"
        description ="""
        Style-LLM-Chat-Single 是使用LLAMA-FACTORY框架上采用自定义古文风格·数据集,使用SFT,RM,PPO进行微调,能够实现在单个模型的基础上实现古文风格的对话。
        项目更多详情见 “https://github.com/YANGCHEN205/style-llm-chat”
        参数详解：
        temperature（温度）: 这个参数用于控制生成文本的随机性。温度较低（比如0.1）会使模型生成的文本比较确定性和一致性强，而较高的温度（比如1.0或更高）会使文本更多样化和随机。
        top_p（也称作nucleus sampling）: 这个参数用于控制在生成每一个词时考虑的概率质量。具体来说，top_p为0.9意味着只考虑累计概率至少达到90%的词。这有助于剔除那些极其罕见的词，使生成的内容更加流畅和合理。
        top_k: 这个参数限制了在生成每个词时考虑的可能词的数量。例如，如果top_k设为50，则每次生成词时只从概率最高的前50个词中选择。
        num_beams（束搜索）: 这个参数用于束搜索，一种用于生成更合理文本的技术。num_beams设置为大于1的值时，模型将探索多种可能的句子组合，选择整体上最有可能（或得分最高）的输出。
        do_sample: 这个布尔参数确定是否在生成文本时进行随机采样。如果设置为True，模型在生成每个词时会基于修改后的概率分布进行抽样，增加文本的多样性。
        return_dict_in_generate: 当这个参数为True时，模型生成函数将返回一个包含更多信息的字典，如生成的词的分数等，而不仅仅是文本。
        output_scores: 如果设置为True，生成的输出将包括预测每个词的分数（通常是概率对数）。这对于理解和分析模型的行为非常有用。
        max_new_tokens: 这个参数限制了生成的最大词（token）数量。这有助于控制生成文本的长度。
        📕对话模型:01ai-Yi1.5-6B-chat         👨‍🦲角色：刘备
        """
        gr.Markdown("<h1 style='text-align: center; margin-bottom: 1rem'>" + title + "</h1>")
        gr.Markdown(description)  # 直接传入 Markdown 文本

        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="对话框")
                msg = gr.Textbox(label="输入框")
                with gr.Row():
                    clear = gr.Button("Clear")
                    retry = gr.Button("retry")
                    submit = gr.Button("Submit", variant="primary")
            with gr.Column(scale=1):
                do_sample = gr.inputs.Checkbox(label="Do sample")
                temperature = gr.components.Slider(minimum=0, maximum=1, value=0.1, label="Temperature")
                top_p = gr.components.Slider(minimum=0, maximum=1, value=0.75, label="Top p")
                top_k = gr.components.Slider(minimum=0, maximum=100, step=1, value=40, label="Top k")
                beams = gr.components.Slider(minimum=1, maximum=1, step=1, value=1, label="Beams(BUG)")
                max_tokens = gr.components.Slider(minimum=1, maximum=512, step=1, value=128, label="Max tokens")
                max_history = gr.components.Slider(minimum=0, maximum=20, step=1, value=10, label="Max history")

        submit.click(evaluate, [msg, do_sample, temperature, top_p, top_k, beams, max_tokens, chatbot, max_history], [msg, chatbot], queue=False)
        clear.click(lambda: "", None, chatbot, queue=False)
        retry.click(lambda v: (v[-1][0] if v else "", v[:-1]), chatbot, [msg, chatbot], queue=False)

    demo.title = title
    demo.launch(server_name=server_name, share=share_gradio)


if __name__ == "__main__":
    fire.Fire(main)