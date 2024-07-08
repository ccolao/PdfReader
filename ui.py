import json

import gradio as gr
import time

import requests
from gradio_pdf import PDF
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import embeddings

import model_utils

#llm = model_utils.llm
connection_args = model_utils.connection_args
embeddings = model_utils.embeddings

def clear():
    return '', None

# 利用langchian构建本地知识库
def create_knowledge_base(file_name, file_path):
    """
    创建知识库
    :param file_path
    :param file_name
    :return:
    """
    try:
        if file_name is None or file_path is None:
            raise gr.Error("创建知识库失败，文件名或文件路径为空！")
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        vector_db = FAISS.from_documents(pages, embeddings)
        vector_db.save_local(file_name)
        gr.Info("知识库创建成功！")
    except Exception as e:
        raise gr.Error(f"创建知识库失败，错误信息：{e}")


def call_model(file_name, temperature, model, prompt, chatbot):
    if not chatbot:
        return chatbot
    query = chatbot[-1][0]
    # 创建请求的JSON数据
    data = {
        "model": model,
        "temperature": temperature,
        "prompt": prompt
    }
    response = requests.post("http://localhost:8000/gradio/call?query={}&file_name={}".format(query, file_name), json=data)
    if response.status_code != 200:
        raise gr.Error(response.msg)
    response = json.loads(response.text)
    content = response.get("data")
    chatbot[-1][1] = content
    return chatbot

def user_input(user_message, chat_history):
    if not user_message:
        return "", chat_history
    if chat_history is None:
        chat_history = []
    return "", chat_history + [[user_message, None]]


def respond(chat_history):
    content = chat_history[-1][1]
    chat_history[-1][1] = ""
    for chat in content:
        chat_history[-1][1] += chat
        time.sleep(0.05)
        yield chat_history


with gr.Blocks(title="PdfReader") as demo:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 编排")
            prompt = gr.Textbox(label="提示词", lines= 5, value = "我希望您能够充当文档评审专家，用下面的内容作为你的知识库，回答用户提出的问题。")
            temperature = gr.Slider(label="temperature", minimum=0.0, maximum=1.0, step=0.1, key="temperature", value=0.2, interactive=True)
            pdf = PDF(label="Upload a PDF", interactive=True)
            pdf.upload(lambda f: f, pdf)
            file_name= gr.Textbox(label="知识库名称")
            create_btn = gr.Button(value="创建本地知识库", variant="primary")
        with gr.Column(scale=2):
            gr.Markdown("### 调试与预览")
            model = gr.Dropdown(choices=["gpt-3.5-turbo", "gpt-4-turbo", "dall-e-3", "whisper"], label="模型", value="gpt-3.5-turbo", interactive=True )
            chatbot = gr.Chatbot()
            query = gr.Textbox()
            with gr.Row():
                submit = gr.Button(value="发送", variant="primary", elem_classes="custom-button")
                clear_btn = gr.Button(value="清空", variant="secondary")
    create_btn.click(fn=create_knowledge_base, inputs=[file_name, pdf])
    submit.click(fn=user_input, inputs=[query, chatbot], outputs=[query, chatbot]).then(
        call_model, [file_name, temperature, model, prompt, chatbot], [chatbot]
    ).then(
        respond, [chatbot], [chatbot]
    )
    # submit.click(fn=call_model, inputs=[query, file_name, temperature, model, prompt, chatbot], outputs=[query, chatbot]).then(
    #     respond, chatbot, chatbot
    # )
    clear_btn.click(fn=clear, outputs=[query, chatbot])


demo.launch(share=True)
