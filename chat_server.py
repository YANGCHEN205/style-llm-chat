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


def main(
    chat_model_config_path: str = "/root/autodl-tmp/LLaMA-Factory/roleplay/chat_model.yaml",
    style_model_config_path: str = "/root/autodl-tmp/LLaMA-Factory/roleplay/style_model.yaml",
    server_name: str = "0.0.0.0",
    share_gradio: bool = False,
):

    ## load model
    with open(chat_model_config_path, 'r', encoding='utf-8') as f:
        chat_model_param = yaml.safe_load(f)
    with open(style_model_config_path, 'r', encoding='utf-8') as f:
        style_model_param = yaml.safe_load(f)

    chat_model = ChatModel(chat_model_param)
    style_model = ChatModel(style_model_param)

    def evaluate(
        input_text,
        chat_model_do_sample,
        chat_model_temperature,
        chat_model_top_p,
        chat_model_top_k,
        chat_model_num_beams,
        chat_model_max_new_tokens,
        chat_model_max_history,
        style_model_do_sample,
        style_model_temperature,
        style_model_top_p,
        style_model_top_k,
        style_model_num_beams,
        style_model_max_new_tokens,
        style_model_max_history,
        chat_model_history=[],
        style_model_history=[],
        **kwargs,
    ):
        # 假设您有 chat_model 和 style_model 这两个模型的实例
        context = input_text
        if chat_model_max_history > 0 and chat_model_history:
            temp_history = sum(chat_model_history, [])
            context = "".join(temp_history[-chat_model_max_history:]) + input_text

        chat_model_prompt = context
        messages = [{"role": "user", "content": chat_model_prompt}]
        chat_model_res = chat_model.chat(
            messages,
            temperature=chat_model_temperature,
            top_p=chat_model_top_p,
            top_k=chat_model_top_k,
            num_beams=chat_model_num_beams,
            do_sample=chat_model_do_sample,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=chat_model_max_new_tokens,
        )

        style_model_prompt = chat_model_res[0].response_text
        messages = [{"role": "user", "content": style_model_prompt}]
        style_model_res = style_model.chat(
            messages,
            temperature=style_model_temperature,
            top_p=style_model_top_p,
            top_k=style_model_top_k,
            num_beams=style_model_num_beams,
            do_sample=style_model_do_sample,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=style_model_max_new_tokens,
        )

        output = style_model_res[0].response_text
        # style_model_prompt = '1111111'
        # output = 'dfsfdfs'
        chat_model_history.append([input_text,style_model_prompt])
        style_model_history.append([input_text, output])
        return "", chat_model_history,style_model_history

    with gr.Blocks() as demo:
        title = "💬Cosplay-ChatGLM-LoRA"
        description = """
        Cosplay-ChatGLM-LoRA 是在 ChatGLM-6B 基础上微调四大名著对话数据得到的，使用的是 LoRA 的方式微调，主要参考了 Stanford Alpaca。更多信息查看 github-LLM_Custome。
        """
        gr.Markdown("<h1 style='text-align: center; margin-bottom: 1rem'>" + title + "</h1>")
        gr.Markdown(description)

        with gr.Row():
            with gr.Column(scale=2):
                chat_model_chatbot = gr.Chatbot(label="chat_model_对话框")


            with gr.Column(scale=1):
                chat_model_do_sample = gr.Checkbox(label="Do sample")
                chat_model_temperature = gr.Slider(minimum=0, maximum=1, value=0.9, label="Temperature")
                chat_model_top_p = gr.Slider(minimum=0, maximum=1, value=0.75, label="Top p")
                chat_model_top_k = gr.Slider(minimum=0, maximum=100, step=1, value=40, label="Top k")
                chat_model_num_beams = gr.Slider(minimum=1, maximum=10, step=1, value=1, label="Beams")
                chat_model_max_new_tokens = gr.Slider(minimum=1, maximum=512, step=1, value=64, label="Max tokens")
                chat_model_max_history = gr.Slider(minimum=0, maximum=20, step=1, value=10, label="Max history")

            with gr.Column(scale=2):
                style_model_chatbot = gr.Chatbot(label="style_model_对话框")

            with gr.Column(scale=1):
                style_model_do_sample = gr.Checkbox(label="Do sample")
                style_model_temperature = gr.Slider(minimum=0, maximum=1, value=0.9, label="Temperature")
                style_model_top_p = gr.Slider(minimum=0, maximum=1, value=0.75, label="Top p")
                style_model_top_k = gr.Slider(minimum=0, maximum=100, step=1, value=40, label="Top k")
                style_model_num_beams = gr.Slider(minimum=1, maximum=10, step=1, value=1, label="Beams")
                style_model_max_new_tokens = gr.Slider(minimum=1, maximum=512, step=1, value=64, label="Max tokens")
                style_model_max_history = gr.Slider(minimum=0, maximum=20, step=1, value=10, label="Max history")
            with gr.Column(scale=1):
                chat_model_msg = gr.Textbox(label="输入框")
                
            submit = gr.Button("提交", variant="primary")
            clear = gr.Button("清除")
            retry = gr.Button("重试")

            submit.click(
                evaluate,
                [
                    chat_model_msg,
                    chat_model_do_sample,
                    chat_model_temperature,
                    chat_model_top_p,
                    chat_model_top_k,
                    chat_model_num_beams,
                    chat_model_max_new_tokens,
                    chat_model_max_history,
                    style_model_do_sample,
                    style_model_temperature,
                    style_model_top_p,
                    style_model_top_k,
                    style_model_num_beams,
                    style_model_max_new_tokens,
                    style_model_max_history,
                    chat_model_chatbot,
                    style_model_chatbot
                ],
                [chat_model_msg, chat_model_chatbot, style_model_chatbot], queue=False
            )

            clear.click(lambda: (None, None,None), None, [chat_model_chatbot, style_model_chatbot], queue=False)
            retry.click(
    lambda v1, v2: (v1[-1][0] if v1 else "", v1[:-1], v2[-1][0] if v2 else ""),  # 第三个输出
    inputs=[chat_model_chatbot],
    outputs=[chat_model_msg, chat_model_chatbot, style_model_chatbot],
    queue=False
)





    demo.title = title
    demo.launch(server_name=server_name, share=share_gradio)


if __name__ == "__main__":
    fire.Fire(main)
