from typing import Any

import uvicorn
from fastapi import FastAPI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
import model_utils

app = FastAPI()


class ModelItem(BaseModel):
    model: str
    temperature: float
    prompt: str

class ModelResponse(BaseModel):
    code: int = 200
    msg: str | None = None
    data: Any | None = None


@app.post("/gradio/call", response_model=ModelResponse)
def call_model(query: str, file_name: str, modelItem: ModelItem):
    """
    调用模型
    :param query: str
    :param file_name: str
    :return: response: ModelItem
    """
    #获取参数
    model = modelItem.model
    temperature = modelItem.temperature
    prompt = modelItem.prompt
    #获取模型
    llm = model_utils.get_chat_model(model, temperature)
    if not file_name:
        return ModelResponse(code=400, msg="请先创建、或选择已经存在的知识库！")
    if model != "gpt-3.5-turbo":
        return ModelResponse(code=400, msg="抱歉，暂不支持该模型，请重新选择！")
    vector_db = FAISS.load_local(file_name, model_utils.embeddings, allow_dangerous_deserialization=True)
    docs = vector_db.similarity_search(query)
    template = """
    {prompt}
    ###
    {text}
    ###

    用户问题：{query}
    """
    promot_template = PromptTemplate.from_template(template)
    llm_chain = promot_template | llm
    content = llm_chain.invoke({"prompt": prompt,"text": "\n".join([doc.page_content for doc in docs]), "query": query}).content
    if not content:
        return ModelResponse(code=400, msg="调用模型失败！")
    return ModelResponse(code=200, msg="sucdess", data=content)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)