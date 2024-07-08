from langchain_openai import ChatOpenAI, OpenAIEmbeddings

def get_chat_model(model="gpt-3.5-turbo", temperature=0):
    return ChatOpenAI(
        openai_api_key= "xxxxx",
        openai_api_base="xxxxx",
        model= model,
        temperature= temperature
    )


embeddings = OpenAIEmbeddings(openai_api_key= "xxxxxx", openai_api_base="xxxxxx")






