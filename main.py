from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import streamlit as st
from langchain.vectorstores.chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import zhipuai

from embedding.call_embedding import get_embedding
from qa_chain.QA_chain_self import QA_chain_self
from qa_chain.Chat_QA_chain_self import Chat_QA_chain_self
import sys
from dotenv import load_dotenv, find_dotenv
import os
from create_db import create_db, load_knowledge_db, create_db_info

##界面
_ = load_dotenv(find_dotenv())
DEFAULT_DB_PATH = "/knowledge_db" # 知识库文件所在路径
DEFAULT_PERSIST_PATH = "/vector_db"     # 向量数据库持久化路径
sys.path.append("/knowledge_db")
sys.path.append("/vector_db")
######界面设计#########

from qa_chain.model_to_llm import model_to_llm


def main():
    st.set_page_config(page_title="ChatGPT Assistant", layout="wide")
    # 定义知识库文件所在路径和向量数据库持久化路径
    answer = None
    st.session_state.persist_path = DEFAULT_PERSIST_PATH
    st.session_state.file_path = DEFAULT_DB_PATH
    # 定义聊天模式选项
    modes = ["None", "qa_chain", "chat_qa_chain"]
    mode_captions = [
        "不使用检索问答的普通模式",
        "不带历史记录的检索问答模式",
        "带历史记录的检索问答模式"
    ]

    # 创建标签页
    tabs = st.tabs(["💬 聊天", "🗒️ 预设", "⚙️ 模型", "🛠️ 功能"])
    from qa_chain.Chat_QA_chain_self import Chat_QA_chain_self

    # 聊天标签页内容
    with tabs[0]:  # 使用索引0来引用第一个标签页，即"💬 聊天"
        st.header("聊天界面")

        # 检查会话状态中是否已经有消息列表，如果没有则初始化
        if 'messages' not in st.session_state:
            st.session_state['messages'] = []

        # 选择对话模式
        selected_method = st.radio(
            "你想选择哪种模式进行对话？",
            options=modes,
            captions=mode_captions,
            key="selected_method"
        )

        # 聊天输入框
        with st.form("chat_form"):
            prompt = st.text_input("Say something:", key="prompt")
            submitted = st.form_submit_button("Submit")

        # 如果提交了聊天输入
        if submitted and prompt:
            if selected_method == "None":
                st.session_state['messages'].append({"role": "user", "text": prompt})
                # 调用 model_to_llm 函数获取 LLM 实例
                llm = model_to_llm(model=st.session_state.selected_model, temperature=st.session_state.temperature,
                                   api_key=st.session_state.temperature)
                # 使用 LLM 实例生成回答
                # 使用get_completion生成answer
                answer = llm.get_completion(prompt, llm, temperature=st.session_state.temperature,
                                            max_tokens=st.session_state.temperature)

                # 将LLM的回答添加到对话历史中
            elif selected_method == "qa_chain":  # 不带历史记录的检索问答模式
                if 'qa_chain' not in st.session_state:
                    # 初始化 QA_chain_self 实例
                    QA = QA_chain_self(
                        model=st.session_state.selected_model,
                        temperature=st.session_state.temperature,
                        top_k=3,
                        file_path=st.session_state.file_path,
                        persist_path=st.session_state.persist_path,
                        api_key=st.session_state.llm_api_key,
                        embedding=st.session_state.embeddings_choice,
                        embedding_key=st.session_state.llm_api_key
                    )
                    answer = QA.answer(question=prompt)
            elif selected_method == "chat_qa_chain":  # 带历史记录的检索问答模式
                if 'chat_qa_chain' not in st.session_state:
                    chat = Chat_QA_chain_self(
                        model=st.session_state.selected_model,
                        temperature=st.session_state.temperature,
                        top_k=3,
                        file_path=st.session_state.file_path,
                        persist_path=st.session_state.persist_path,
                        api_key=st.session_state.llm_api_key,
                        embedding=st.session_state.embeddings_choice,
                        embedding_key=st.session_state.llm_api_key
                    )
                    answer = chat.answer(question=prompt)
        if answer is not None:
            # 将LLM的回答添加到对话历史中
            st.session_state.messages.append({"role": "assistant", "text": answer})
    # 显示整个对话历史
    for message in st.session_state['messages']:
        if message["role"] == "user":
            st.write(f"用户: {message['text']}")
        elif message["role"] == "assistant":
            st.write(f"助手: {message['text']}")


    with tabs[1]:
        st.header("预设")

    with tabs[2]:
        LLM_MODEL_DICT = {
            "openai": ["gpt-3.5-turbo", "gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-0613", "gpt-4", "gpt-4-32k"],
            "wenxin": ["ERNIE-Bot", "ERNIE-Bot-4", "ERNIE-Bot-turbo"],
            "xinhuo": ["Spark-1.5", "Spark-2.0"],
            "zhipuai": ["chatglm_pro", "chatglm_std", "chatglm_lite"]
        }

        st.header("模型设置")

        # 初始化温度滑块
        if "temperature" not in st.session_state:
            st.session_state.temperature = 0.7
        st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, st.session_state.temperature)

        # 初始化平台选择
        if "selected_platform" not in st.session_state:
            st.session_state.selected_platform = "openai"

        # 平台选择
        selected_platform = st.selectbox(
            "选择平台：",
            options=list(LLM_MODEL_DICT.keys()),
            index=list(LLM_MODEL_DICT.keys()).index(st.session_state.selected_platform)
        )

        # 当选择的 platform 发生变化时，重新初始化 selected_model
        if selected_platform != st.session_state.selected_platform:
            st.session_state.selected_platform = selected_platform
            st.session_state.selected_model = LLM_MODEL_DICT[selected_platform][0]  # 更新为新平台的默认模型

        # 初始化模型选择
        if "selected_model" not in st.session_state:
            st.session_state.selected_model = LLM_MODEL_DICT[selected_platform][0]

        # 模型选择
        selected_model = st.selectbox(
            "选择模型：",
            options=LLM_MODEL_DICT[selected_platform],
            index=LLM_MODEL_DICT[selected_platform].index(st.session_state.selected_model),
            key="selected_model"
        )

        # 更新 selected_model 到 session_state
        if selected_model != st.session_state.selected_model:
            st.session_state.selected_model = selected_model

        # LLM API密钥输入
        api_key = st.text_input("LLM API Key", type="password")

        # 显示当前选择
        st.write(f"当前选择的平台：{st.session_state.selected_platform}")
        st.write(f"当前选择的模型：{st.session_state.selected_model}")

        # 保存 API 密钥
        if api_key:
            st.session_state.llm_api_key = api_key
            st.success("API密钥已保存。")
        else:
            st.warning("请提供LLM API密钥以使用模型功能。")

    with tabs[3]:
        st.header("功能设置")

        # 清除聊天记录的按钮
        clear = st.button("清除聊天记录")

        # 如果用户点击了清除聊天记录按钮
        if clear:
            # 检查会话状态中是否有聊天记录，如果有则清除
            if 'messages' in st.session_state:
                st.session_state.messages = []
                st.success("聊天记录已清除！")
            else:
                st.info("没有聊天记录可清除。")



    with st.sidebar:
        st.header("知识库设置")

        # 上传文件
        uploaded_file = st.file_uploader("上传知识库文件", type=["txt", "pdf", "md", "docx"])

        if uploaded_file is not None:
                # 构建文件路径，确保文件名没有冲突
            if not os.path.exists(DEFAULT_DB_PATH):
                os.makedirs(DEFAULT_DB_PATH)  # 如果文件夹不存在，则创建文件夹

            file_path = os.path.join(DEFAULT_DB_PATH, uploaded_file.name)
            file_bytes = uploaded_file.read()
                # 保存上传的文件

            with open(file_path, 'wb') as f:
                f.write(file_bytes)
            st.success(f"文件 {uploaded_file.name} 已成功上传！")

        st.session_state.embeddings_choice = st.selectbox("选择 Embedding 模型", ["openai", "m3e", "zhipuai"])


if __name__ == "__main__":
    main()
