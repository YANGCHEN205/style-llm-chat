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

    # ## load model
    # with open(chat_model_config_path, 'r', encoding='utf-8') as f:
    #     chat_model_param = yaml.safe_load(f)
    # with open(style_model_config_path, 'r', encoding='utf-8') as f:
    #     style_model_param = yaml.safe_load(f)

    # chat_model = ChatModel(chat_model_param)
    # style_model = ChatModel(style_model_param)


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

        # context = input_text
        # if max_history > 0 and history:
        #     tem_history = sum(history, [])
        #     context = "".join(tem_history[-max_history:]) + input_text
        # chat_model_prompt = context


        # messages = [{"role": "user", "content": chat_model_prompt}]
        # chat_model_res = chat_model.chat(messages)
        # style_model_prompt = chat_model_res[0].response_text
        # messages = [{"role": "user", "content": style_model_prompt}]
        # style_model_res = style_model.chat(messages)
        
        
        # print("style reponse: ", style_model_res[0].response_text)
        # print("chat reponse: ",chat_model_res[0].response_text)
        # with torch.no_grad():
            # generation_output = model.generate(
            #     input_ids=input_ids,
            #     generation_config=generation_config,
            #     return_dict_in_generate=True,
            #     output_scores=True,
            #     max_new_tokens=max_new_tokens,
            # )
        # s = generation_output.sequences[0]
        # output = tokenizer.decode(s)[len(context):]
        # output = style_model_res[0].response_text
        output = 'werwerwer'
        history.append([input_text, output])
        for line in history:
            print(line[0])
            print(line[1])
        return "", history

    with gr.Blocks() as demo:
        title = "💬Cosplay-ChatGLM-LoRA"
        description = """
        Cosplay-ChatGLM-LoRA 是在 [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) 基础上微调四大名著对话数据得到的，使用的是 LoRA 的方式微调，主要参考了 [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)。更多信息查看 [github-LLM_Custome](https://github.com/wellinxu/LLM_Custome)。
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