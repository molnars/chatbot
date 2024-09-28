import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import base64
import warnings
import os
import re
import fitz
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.schema.document import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from streamlit_pdf_viewer import pdf_viewer
from pathlib import Path
import base64
from langchain_community.embeddings import OllamaEmbeddings
from data_extraction_rag_setup import extract_n_create_rag
import streamlit.components.v1 as components
from langchain_openai import OpenAIEmbeddings
from pymilvus import connections, utility
from langchain_community.vectorstores import Milvus
from streamlit_feedback import streamlit_feedback
from ml_module import *
import pyperclip
import uuid
from langchain_community.embeddings import HuggingFaceEmbeddings
warnings.filterwarnings('ignore')

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg():
    bin_str = get_base64_of_bin_file("./mascot.png")
    page_bg_img = '''
        <style>
        .stApp{
        background-image: url("data:image/png;base64,%s");
        background-repeat: no-repeat;
        background-size: 600px 600px;
        background-position: center bottom;
        }
        </style>
        ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return
    
def logo_heading():
    APP_TITLE = "Nawras" #os.getenv('APP_TITLE', 'Document AI 🤖')
    nagarro_IMAGE = "./maskot.png"
    st.markdown(
        """
        <style>
        .container {
            display: flex;
        }
        .logo-text {
            font-weight:0 !important;
            font-size:35px !important;
            color: #D046BD !important;
            padding-top: 10px !important;
            padding-left: 30px !important;
        }
        .logo-text2 {
            font-weight:0 !important;
            font-size:15px !important;
            color: #D046BD !important;
            padding-top: 30px !important;
            padding-left: 10px !important;
        }
        .logo-img {
            float:left;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        f"""
                <div class="container">
                    <img class="logo-img" src="data:image/png;base64,{base64.b64encode(open(nagarro_IMAGE, "rb").read()).decode()}" 
                    width="60" height="60">
                    <p class="logo-text">{APP_TITLE}</p>
                </div>
                """,
        unsafe_allow_html=True
    )


def show_image(img_html):
    st.markdown(
        """
        <style>
        .container {
            display: flex;
        }
        .logo-img {
            float:left;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        f"""
                <div class="container">
                    {img_html}
                    width="500" height="500">
                </div>
                """,
        unsafe_allow_html=True
    )


def clear_history():
    if 'messages' in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

def on_change():
    if 'messages' in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]
    if "option" in st.session_state:
        del st.session_state.option

def setting_setup():
    admin_pass = None
    option_val = None
    with st.expander("Use application as Admin"):
        admin_pass = st.text_input("Admin Password", type='password')

    milvus_host_tmp = os.getenv("MILVUS_HOST")
    milvus_port_tmp = os.getenv("MILVUS_PORT")
    embedding_model_tmp = os.getenv("EMBEDDING_MODEL_NAME")

    if milvus_host_tmp is None:
        milvus_host_tmp = "localhost"
    if milvus_port_tmp is None:
        milvus_port_tmp = 19530

    milvus_host = st.text_input("Give Milvus Host details",value=milvus_host_tmp)
    milvus_port = st.text_input("Give Milvus Port number",value=milvus_port_tmp)
    llm_choices = ["Choose an option", "OpenAI GPT4o", "Llama 3.1"]

    if option_val is not None:
        index_val = llm_choices.index(option_val)
    else:
        index_val = 0

    option = st.selectbox("Which LLM do you want to use?",
                          options=llm_choices,
                          index=index_val,
                          on_change=on_change)
    if option:
        option_val = option
        if "option_val" not in st.session_state:
            st.session_state.option_val = option
    if option_val == "OpenAI GPT4o":
        llm_api_key = st.text_input("Give the OpenAI API key", type='password')
    else:
        llm_api_key = None

    sub = st.button("Submit")
    if sub:
        st.session_state.session_id_u = str(uuid.uuid4())
        if admin_pass:
            st.session_state.admin_pass = admin_pass
        if milvus_host:
            st.session_state.milvus_host = milvus_host
        if milvus_port:
            st.session_state.milvus_port = milvus_port
        if llm_api_key:
            st.session_state.llm_api_key = llm_api_key

        if option == "OpenAI GPT4o" and "llm_api_key" in st.session_state and st.session_state.llm_api_key!="":
            st.session_state.option = option
            st.session_state.embedding_model = "text-embedding-3-small"
        elif option == "Llama 3.1":
            st.session_state.option = option
            if embedding_model_tmp is not None:
                st.session_state.embedding_model = embedding_model_tmp
            else:
                st.session_state.embedding_model = "nomic-embed-text"

        if "option" in st.session_state:
            st.write(f"Embedding model - '{st.session_state.embedding_model}' will be used.")
            st.success("Configuration is complete :+1:")
        else:
            st.error("Setup incomplete")

def get_string(in_dict, str_val, entry_pt):
    for k, v in in_dict.items():
        if type(v) == list:
            if type(v[0]) == dict:
                entry_pt_tmp = entry_pt + 3
                for i in v:
                    str_val = get_string(i, str_val, entry_pt_tmp)
            else:
                spc = "&ensp;" * entry_pt
                str_tmp = "\n".join(v)
                str_val = str_val + f"<tr><td style='width:30%'>{spc}{k}</td><td>{spc}{str_tmp}</td></tr>"
        elif type(v) == dict and len(v) > 0:
            str_val = str_val + f"<tr><td style='width:30%'><b>{k}</b></td></tr>"
            entry_pt_tmp = entry_pt + 3
            str_val = get_string(v, str_val, entry_pt_tmp)
        elif type(v) == dict and len(v) <= 0:
            str_val = str_val + f"<tr><td style='width:30%'><b>{k}</b></td></tr>"
        else:
            spc = "&ensp;" * entry_pt
            str_val = str_val + f"<tr><td style='width:30%'>{spc}{k}</td><td>{spc}{v}</td></tr>"
    return str_val

def get_string_v(in_dict, str_val, entry_pt):
    for k, v in in_dict.items():
        if type(v) == list:
            if type(v[0]) == dict:
                entry_pt_tmp = entry_pt + 2
                for i in v:
                    str_val = get_string_v(i, str_val, entry_pt_tmp)
            else:
                spc = "\t" * entry_pt
                str_tmp = f"\n{spc}* ".join(v)
                str_val = str_val + f"\n{spc}{k} : \n{spc}* {str_tmp} \n"
        elif type(v) == dict and len(v) > 0:
            str_val = str_val + f"{k} :\n"
            entry_pt_tmp = entry_pt + 2
            str_val = get_string_v(v, str_val, entry_pt_tmp)
        elif type(v) == dict and len(v) <= 0:
            str_val = str_val + f"\n{k} :\n"
        else:
            spc = "\t" * entry_pt
            if "table" in k or "text" == k or "additional_text" == k :
                str_val = str_val + f"{spc}{v} \n"
            else:
                str_val = str_val + f"{spc}{k} : {v} \n"
    return str_val

def save_fb_csv(llm_model_name, query_asked, answer, reference_file, feedback):
    session_id  = st.session_state.session_id_u
    df_r = {}
    df_r["session_id"] = [session_id]
    df_r["LLM_Model"] = [llm_model_name]
    df_r["Query"] = [query_asked]
    df_r["Answer"] = [answer]
    df_r["Reference_File"] = [" | ".join(reference_file)]
    df_r["Feedback"] = ["Not Provided" if feedback is None or feedback == "" else feedback["score"]]
    df_r_pd = pd.DataFrame(df_r)
    csv_fb_path = "./usage_records/query_response_feedback.csv"
    my_file = Path(csv_fb_path)
    if my_file.is_file():
        df_tmp = pd.read_csv(csv_fb_path)
        tmp_dict_list = df_tmp.to_dict("records")
        update_flag = False
        for j,i in enumerate(tmp_dict_list):
            if (i["session_id"] == session_id and i["LLM_Model"] == llm_model_name
                    and i["Query"] == query_asked and i["Feedback"] == "Not Provided"):
                tmp_dict_list[j]["Feedback"] = ["Not Provided" if feedback is None or feedback == "" else feedback["score"]][0]
                update_flag = True
                break
        if update_flag:
            update_flag = False
            df_r_o = pd.DataFrame(tmp_dict_list)
            os.remove(csv_fb_path)
            df_r_o.to_csv(csv_fb_path, index=False, header=True)
        else:
            df_r_pd.to_csv(csv_fb_path, mode='a', index=False, header=False)
    else:
        df_r_pd.to_csv(csv_fb_path, index=False, header=True)


def save_rag_context(llm_model_name, query_asked, context_data, score, document_type):
    session_id  = st.session_state.session_id_u
    df_r = {}
    df_r["session_id"] = [session_id]
    df_r["LLM_Model"] = [llm_model_name]
    df_r["Query"] = [query_asked]
    df_r["Context_Data"] = [context_data]
    df_r["Document_Type"] = [document_type]
    df_r["Score"] = [score]
    df_r_pd = pd.DataFrame(df_r)
    csv_fb_path = "./usage_records/query_context_rag.csv"
    my_file = Path(csv_fb_path)
    if my_file.is_file():
        df_r_pd.to_csv(csv_fb_path, mode='a', index=False, header=False)
    else:
        df_r_pd.to_csv(csv_fb_path, index=False, header=True)


def plot_bar_by_class(df, col_name):
    class_counts = df[col_name].value_counts()
    class_percentages = (class_counts / len(df)) * 100

    fig, ax = plt.subplots(figsize=(3, 3.5))
    bars = ax.bar(class_percentages.index, class_percentages, color='skyblue', edgecolor='black')

    # Adding labels and title
    plt.title('', fontsize=6, fontweight='bold')
    plt.xlabel(col_name, fontsize=6)
    plt.ylabel('Distribution Percentage', fontsize=6)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)

    # Adding grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Display the percentage on the top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=5, color='blue')

    # Adding a footer note
    plt.figtext(0.99, 0.01, '', ha='right', fontsize=6, style='italic')

    # Adding a tight layout so everything fits without overlapping
    plt.tight_layout()
    return plt


def usage_report():
    csv_fb_path = "./usage_records/query_response_feedback.csv"
    csv_rag_path = "./usage_records/query_context_rag.csv"
    my_file = Path(csv_fb_path)
    my_file_r = Path(csv_rag_path)
    if my_file.is_file():
        df_tmp = pd.read_csv(csv_fb_path)
        st.dataframe(df_tmp)

        df_tmp.loc[df_tmp["Feedback"] == "👎", "Feedback"] = "Thumbs Down"
        df_tmp.loc[df_tmp["Feedback"] == "👍", "Feedback"] = "Thumbs Up"
        plt = plot_bar_by_class(df_tmp, "Feedback")
        st.pyplot(plt,use_container_width=False)

    if my_file_r.is_file():
        df_tmp_r = pd.read_csv(csv_rag_path)
        st.dataframe(df_tmp_r)

    if not my_file.is_file() and not my_file_r.is_file():
        st.warning("No feedback received yet...")


def prompt_edit():
    model_option = None
    if "option" in st.session_state:
        model_option = st.session_state.option
    if model_option == "OpenAI GPT4o":
        with st.form(key="Form :", clear_on_submit=False):
            rag_prompt = ""
            general_prompt = ""
            if "txt_rag_prompt" in st.session_state:
                rag_prompt = st.session_state.txt_rag_prompt
            else:
                rag_prompt = "Answer only the query above in proper detailed manner, possibly in bullet points with proper headings in text format."

            if "txt_general_prompt" in st.session_state:
                general_prompt = st.session_state.txt_general_prompt
            else:
                general_prompt = "Answer only the query above in proper detailed manner in text format."

            txt_rag_prompt = st.text_area(
                                            label = "Prompt for RAG",
                                            value= rag_prompt
                                            )
            txt_general_prompt = st.text_area(
                                                label="General Prompt",
                                                value=general_prompt
                                            )
            Submit = st.form_submit_button(label='Save')
        if Submit:
            st.session_state.txt_rag_prompt = txt_rag_prompt
            st.session_state.txt_general_prompt = txt_general_prompt
            st.success("Prompts are saved")
        else:
            st.session_state.txt_rag_prompt = rag_prompt
            st.session_state.txt_general_prompt = general_prompt
    elif model_option == "Llama 3.1":
        with st.form(key="Form :", clear_on_submit=False):
            rag_prompt = ""
            general_prompt = ""
            if "txt_rag_prompt" in st.session_state:
                rag_prompt = st.session_state.txt_rag_prompt
            else:
                rag_prompt = "Answer only the query above in proper detailed manner, possibly in bullet points with proper headings in text format."

            if "txt_general_prompt" in st.session_state:
                general_prompt = st.session_state.txt_general_prompt
            else:
                general_prompt = "Answer only the query above in proper detailed manner in text format."

            txt_rag_prompt = st.text_area(
                label="Prompt for RAG",
                value=rag_prompt
            )
            txt_general_prompt = st.text_area(
                label="General Prompt",
                value=general_prompt
            )
            Submit = st.form_submit_button(label='Save')
        if Submit:
            st.session_state.txt_rag_prompt = txt_rag_prompt
            st.session_state.txt_general_prompt = txt_general_prompt
            st.success("Prompts are saved")
        else:
            st.session_state.txt_rag_prompt = rag_prompt
            st.session_state.txt_general_prompt = general_prompt
    else:
        st.warning("Please configure the application in settings...")


def search_ai():
    milvus_host = None
    milvus_port = None
    model_option = None
    llm_api_key = None
    embedding_model_option = None
    embedding_model = None
    vectorstore_r = None

    if "option" in st.session_state:
        model_option = st.session_state.option
    if "milvus_host" in st.session_state:
        milvus_host = st.session_state.milvus_host
    if "milvus_port" in st.session_state:
        milvus_port = st.session_state.milvus_port

    if model_option == "OpenAI GPT4o" and "llm_api_key" in st.session_state:
        llm_api_key = st.session_state.llm_api_key
    if "embedding_model" in st.session_state:
        embedding_model_option = st.session_state.embedding_model

    if model_option == "OpenAI GPT4o" and llm_api_key != "" and llm_api_key is not None:
        embedding_model = OpenAIEmbeddings(model=embedding_model_option, openai_api_key=llm_api_key)
    elif model_option == "Llama 3.1":
        OLLAMA_FLAG = str(os.getenv('OLLAMA_FLAG_EMBED', True))
        if OLLAMA_FLAG == "True":
            INFERENCE_SERVER_URL = os.getenv('EMBEDDING_SERVER_URL')
            model_name_embd = os.getenv('EMBEDDING_MODEL_NAME', 'nomic-embed-text')
            embedding_model = OllamaEmbeddings(model=model_name_embd, base_url=INFERENCE_SERVER_URL)
        else:
            model_kwargs = {'trust_remote_code': True}
            embedding_model = HuggingFaceEmbeddings(
                                                    model_name="nomic-ai/nomic-embed-text-v1",
                                                    model_kwargs=model_kwargs,
                                                    show_progress=False
                                                )

    my_file = Path("./usage_records/record_rag_index.csv")
    list_doc_types = {}
    if my_file.is_file():
        df_tmp = pd.read_csv("./usage_records/record_rag_index.csv")
        tmp_dict_list = df_tmp.to_dict("records")

        for i in tmp_dict_list:
            if  model_option == i["model"]:
                list_doc_types[str(i["doc_type"]).title()] = i["collection_index"]


    bt = None
    lst_type_options_search = []
    with st.sidebar:
        lst_type_options_search = list(list_doc_types.keys())
        lst_type_options = lst_type_options_search + ["Talk to LLM"]
        option_val_r = ""
        index_selected = 0
        if "option_a" in st.session_state:
            option_val_r = st.session_state.option_a
            index_selected = lst_type_options.index(option_val_r)
        option_a = st.radio(label ='Select the document type for search',options= lst_type_options,
                            index= index_selected)
        bt = st.button("lets get started")
        if bt:
            st.session_state.option_a = option_a
            st.session_state.bt = bt
            st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

    if "bt" in st.session_state:
        if model_option is None:
            st.warning("Please configure the application in settings...")
            del st.session_state.bt
        elif model_option is not None and len(lst_type_options_search) == 0:
            with st.sidebar:
                st.warning(f"There is no index found in RAG, based on the selected model '{model_option}'. Please contact System Admin for assistance.")

        index_option_selected = st.session_state.option_a
        if model_option is not None and index_option_selected != "Talk to LLM":
            collection_name_r = None
            if milvus_host != "":
                if list_doc_types is not None and len(list_doc_types) > 0 and index_option_selected is not None:
                    collection_name_r = list_doc_types[index_option_selected]
                    MILVUS_HOST = os.getenv('MILVUS_HOST')
                    MILVUS_PORT = os.getenv('MILVUS_PORT')
                    MILVUS_USERNAME = os.getenv('MILVUS_USERNAME')
                    MILVUS_PASSWORD = os.getenv('MILVUS_PASSWORD')
                    MILVUS_TOKEN = os.getenv('MILVUS_TOKEN')

                    if MILVUS_USERNAME != "None" and MILVUS_PASSWORD != "None":
                        print("With password")
                        connection_args_dict = {"host": MILVUS_HOST, "port": MILVUS_PORT, "user": MILVUS_USERNAME,
                                                "password": MILVUS_PASSWORD, "timeout": 300}
                    else:
                        print("Without password")
                        connection_args_dict = {"host": MILVUS_HOST, "port": milvus_port ,
                                                "token": MILVUS_TOKEN, "timeout": 300}
                        #connection_args_dict = {"host": MILVUS_HOST, "port": MILVUS_PORT , "timeout": 300}
                    vectorstore_r = Milvus.from_documents(documents=[],
                                                           embedding=embedding_model,
                                                           collection_name=collection_name_r,
                                                           connection_args=connection_args_dict)
            with st.container(height=75, border=False):
                if "messages" not in st.session_state.keys():
                    st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]
                    for message in st.session_state.messages:
                        if "image" not in message:
                            with st.chat_message(message["role"]):
                                st.write(message["content"])
                else:
                    message = st.session_state.messages[0]
                    if "image" not in message:
                        with st.chat_message(message["role"]):
                            st.write(message["content"])
                    if st.session_state.messages[-1]["role"] == "assistant":
                        if "fb_k" in st.session_state:
                            st.session_state.messages[-1]["feedback"] = st.session_state.fb_k
                            query_asked = ""
                            answer = st.session_state.messages[-1]["content"]
                            flag_update = True
                            if "file_namer" in st.session_state.messages[-1]:
                                reference_file = st.session_state.messages[-1]["file_namer"]
                            else:
                                flag_update = False
                            if "question_asked" in st.session_state:
                                query_asked = st.session_state.question_asked
                                del st.session_state.question_asked
                            if flag_update:
                                save_fb_csv(model_option, query_asked, answer, reference_file, st.session_state.fb_k)
                            del st.session_state.fb_k
                    countr = -1
                    for message in st.session_state.messages:
                        countr = countr + 1
                        if message["role"] == "assistant":
                            if f'fb_k_{countr}' in st.session_state:
                                st.session_state.messages[countr]["feedback"] = st.session_state[f'fb_k_{countr}']
                                answer = message["content"]
                                reference_file = message["file_namer"]
                                query_asked = st.session_state.messages[countr-1]["content"]
                                save_fb_csv(model_option, query_asked, answer, reference_file, st.session_state[f'fb_k_{countr}'])

            if prompt := st.chat_input("Please ask your query here..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.question_asked = prompt
                with st.chat_message("user"):
                    st.write(prompt)
            response = ""
            if "messages" in st.session_state.keys():
                if st.session_state.messages[-1]["role"] != "assistant":
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            MAX_RETRIEVED_DOCS = int(os.getenv('MAX_RETRIEVED_DOCS', 7))
                            search_result = vectorstore_r.similarity_search_with_score(prompt,k= MAX_RETRIEVED_DOCS)
                            context = []
                            for y in search_result:
                                i = y[0]
                                score = y[1]
                                context_data = i.page_content + f"""\n Table_Data = {i.metadata["table_content"]} \n page_number = {i.metadata["page_num"]} \n file_name = {i.metadata["source_file_name"]} \n page_image = {i.metadata["page_image"]} """
                                context.append(context_data)
                                save_rag_context(model_option, prompt, context_data, score, index_option_selected)

                            if "txt_rag_prompt" in st.session_state:
                                rag_prompt = st.session_state.txt_rag_prompt
                            else:
                                rag_prompt = "Answer only the query above in proper detailed manner, possibly in bullet points with proper headings in text format."

                            if model_option == "OpenAI GPT4o":
                                response_dict = answer_query_openai(llm_api_key, prompt, context, rag_prompt)
                            else:
                                response_dict = answer_query_llama(prompt, context, rag_prompt)
                            response = response_dict["answer"]
                            img_file_data = response_dict["source_details"]

                            file_data_list = img_file_data["file_name"]
                            if "NA" in file_data_list:
                                file_data = "NA"
                                file_name_r = "NA"
                            elif type(file_data_list) == list:
                                file_data = ["./uploaded_files/" + i for i in file_data_list]
                                file_name_r = file_data_list
                            elif type(file_data_list) == str:
                                if "," in file_data_list:
                                    file_data_list_tmp = file_data_list.split(",")
                                    file_data = ["./uploaded_files/" + i.strip() for i in file_data_list_tmp]
                                    file_name_r = file_data_list_tmp
                                else:
                                    file_name_r = file_data_list
                                    file_data = "./uploaded_files/" + file_data_list
                                if "," in file_name_r:
                                    file_name_r = file_name_r.split(",")[0].strip()
                                    file_data = "./uploaded_files/" + file_name_r

                            col1,col2 = st.columns([1,1])
                            response_str = response.replace("**", "")
                            feedback = None
                            img_data = img_file_data["page_image"]
                            if "NA" in img_data:
                                response_str = "Sorry, could not answer."
                                file_data = "NA"
                                img_data = "NA"
                            elif type(img_data) == str:
                                if "," in img_data:
                                    img_data_tmp = img_data.split(",")
                                    img_data = [ i.strip() for i in img_data_tmp]

                            with st.container():
                                with col1:
                                    st.write(response_str)
                                    if response_str != "Sorry, could not answer.":
                                        if st.button("📋"):
                                            pyperclip.copy(response_str)
                                            st.success('Text copied successfully!')

                                with col2:
                                    with st.expander(":violet[Reference Document Pages...]"):
                                        if "NA" not in img_data:
                                            st.image(img_data, use_column_width=True)
                                    if "NA" not in file_data:
                                        for i,j in enumerate(file_data):
                                            with open(j, "rb") as pdf_file:
                                                PDFbyte = pdf_file.read()
                                            st.download_button(label=f":violet[Download Source File - {i+1}]",
                                                               data=PDFbyte,
                                                               file_name=f"{file_name_r[i]}",
                                                               key = f"button_r_{i}",
                                                               mime='application/octet-stream')

                                feedback = streamlit_feedback(feedback_type="thumbs", key='fb_k'
                                                              , on_submit='streamlit_feedback')

                            message = {"role": "assistant", "content": response_str,"image": img_data,
                                       "file_name": file_data,"file_namer": file_name_r,"feedback":None}
                            st.session_state.messages.append(message)
                elif len(st.session_state.messages) > 1:
                    count = "else_msg"
                    if st.session_state.messages[-1]["role"] == "assistant":
                        if "fb_k_e" in st.session_state:
                            st.session_state.messages[-1]["feedback"] = st.session_state.fb_k_e
                            answer = st.session_state.messages[-1]["content"]
                            reference_file = st.session_state.messages[-1]["file_namer"]
                            query_asked = st.session_state.messages[-2]["content"]
                            save_fb_csv(model_option, query_asked, answer, reference_file, st.session_state.fb_k_e)
                            del st.session_state.fb_k_e

                    message = st.session_state.messages[-1]
                    message_u = st.session_state.messages[-2]

                    if message_u["role"] != "assistant":
                        with st.chat_message(message_u["role"]):
                            st.write(message_u["content"])
                    if message["role"] == "assistant":
                        with st.chat_message(message["role"]):
                            col1, col2 = st.columns([1, 1])
                            with st.container():
                                with col1:
                                    resp_txt = message["content"]
                                    st.write(resp_txt)
                                    if resp_txt != "Sorry, could not answer.":
                                        if st.button("📋", key=f"bt_{count}", use_container_width=False):
                                            pyperclip.copy(resp_txt)
                                            st.success('Text copied successfully!')
                                with col2:
                                    img_data = message["image"]
                                    if "NA" not in img_data:
                                        with st.popover(":violet[Reference Document Pages...]", use_container_width=True):
                                            img_data = message["image"]
                                            if "NA" not in img_data:
                                                st.image(message["image"], use_column_width=True)

                                        file_data = message["file_name"]
                                        file_name_r = message["file_namer"]
                                        if "NA" not in file_data:
                                            for i,j in enumerate(file_data):
                                                with open(j, "rb") as pdf_file:
                                                    PDFbyte = pdf_file.read()
                                                st.download_button(label=f":violet[Download Source File - {i+1}]",
                                                                   data=PDFbyte,
                                                                   file_name=f"{file_name_r[i]}",
                                                                   key = f"button_r_{i}_{count}",
                                                                   mime='application/octet-stream')
                                if "feedback" in message:
                                    if message['feedback'] is None:
                                        feedback = streamlit_feedback(feedback_type="thumbs", key='fb_k_e', on_submit='streamlit_feedback')


            if "messages" in st.session_state and len(st.session_state.messages) > 1:
                with st.container(border=False):
                    with st.expander("Conversation History...", expanded=False):
                        if "messages" in st.session_state.keys():
                            count = -1
                            for message in st.session_state.messages:
                                count = count + 1
                                if "image" not in message:
                                    with st.chat_message(message["role"]):
                                        st.write(message["content"])
                                else:
                                    with st.chat_message(message["role"]):
                                        col1, col2 = st.columns([1, 1])
                                        with col1:
                                            resp_txt = message["content"]
                                            st.write(resp_txt)
                                            if resp_txt != "Sorry, could not answer.":
                                                if st.button("📋", key=f"bt_{count}", use_container_width=False):
                                                    pyperclip.copy(resp_txt)
                                                    st.success('Text copied successfully!')
                                        with col2:
                                            img_data = message["image"]
                                            if "NA" not in img_data:
                                                with st.popover(":violet[Reference Document Pages...]",use_container_width=True):
                                                    img_data = message["image"]
                                                    if "NA" not in img_data:
                                                        st.image(message["image"], use_column_width=True)

                                                file_data = message["file_name"]
                                                file_name_r = message["file_namer"]
                                                if "NA" not in file_data:
                                                    for i, j in enumerate(file_data):
                                                        with open(j, "rb") as pdf_file:
                                                            PDFbyte = pdf_file.read()
                                                        st.download_button(label=f":violet[Download Source File - {i+1}]",
                                                                           data=PDFbyte,
                                                                           file_name=f"{file_name_r[i]}",
                                                                           key=f"button_key_r_{i}_{count}",
                                                                           mime='application/octet-stream')
                                                if "feedback" in message:
                                                    if message['feedback'] is None:
                                                        feedback = streamlit_feedback(feedback_type="thumbs", key=f'fb_k_{count}'
                                                                                      , on_submit='streamlit_feedback')
                                                    else:
                                                        st.write(
                                                            f"Thank you for the feedback. Your feedback was - {message['feedback']['score']}")
        elif model_option is not None and index_option_selected == "Talk to LLM":
            with st.container(height=75, border=False):
                if "messages" not in st.session_state.keys():
                    st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]
                    for message in st.session_state.messages:
                        if "image" not in message:
                            with st.chat_message(message["role"]):
                                st.write(message["content"])
                else:
                    message = st.session_state.messages[0]
                    with st.chat_message(message["role"]):
                        st.write(message["content"])
                    if st.session_state.messages[-1]["role"] == "assistant":
                        if "fb_k" in st.session_state:
                            st.session_state.messages[-1]["feedback"] = st.session_state.fb_k
                            query_asked = ""
                            answer = st.session_state.messages[-1]["content"]
                            if "question_asked" in st.session_state:
                                query_asked = st.session_state.question_asked
                                del st.session_state.question_asked
                            save_fb_csv(model_option, query_asked, answer, [], st.session_state.fb_k)
                            del st.session_state.fb_k
                    countr = -1
                    for message in st.session_state.messages:
                        countr = countr + 1
                        if message["role"] == "assistant":
                            if f'fb_k_{countr}' in st.session_state:
                                st.session_state.messages[countr]["feedback"] = st.session_state[f'fb_k_{countr}']
                                answer = message["content"]
                                reference_file = []
                                query_asked = st.session_state.messages[countr-1]["content"]
                                save_fb_csv(model_option, query_asked, answer, reference_file, st.session_state[f'fb_k_{countr}'])

            if prompt := st.chat_input("Please ask your query here..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.question_asked = prompt
                with st.chat_message("user"):
                    st.write(prompt)
            response = ""
            if "messages" in st.session_state.keys():
                if st.session_state.messages[-1]["role"] != "assistant":
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            if "txt_general_prompt" in st.session_state:
                                general_prompt = st.session_state.txt_general_prompt
                            else:
                                general_prompt = "Answer only the query above in proper detailed manner in text format."

                            if model_option == "OpenAI GPT4o":
                                response = answer_general_query_openai(llm_api_key, prompt, general_prompt)
                            else:
                                response = answer_general_query_llama(prompt, general_prompt)

                            response_str = response.replace("**", "")
                            feedback = None
                            with st.container():
                                st.write(response_str)
                                if st.button("📋"):
                                    pyperclip.copy(response_str)
                                    st.success('Text copied successfully!')

                                feedback = streamlit_feedback(feedback_type="thumbs", key='fb_k'
                                                              , on_submit='streamlit_feedback')

                                message = {"role": "assistant", "content": response_str,"feedback":None}
                                st.session_state.messages.append(message)
                elif len(st.session_state.messages) > 1:
                    count = "else_msg"
                    if st.session_state.messages[-1]["role"] == "assistant":
                        if "fb_k_e" in st.session_state:
                            st.session_state.messages[-1]["feedback"] = st.session_state.fb_k_e
                            answer = st.session_state.messages[-1]["content"]
                            reference_file = []
                            query_asked = st.session_state.messages[-2]["content"]
                            save_fb_csv(model_option, query_asked, answer, reference_file, st.session_state.fb_k_e)
                            del st.session_state.fb_k_e

                    message = st.session_state.messages[-1]
                    message_u = st.session_state.messages[-2]

                    if message_u["role"] != "assistant":
                        with st.chat_message(message_u["role"]):
                            st.write(message_u["content"])
                    if message["role"] == "assistant":
                        with st.chat_message(message["role"]):
                            with st.container():
                                resp_txt = message["content"]
                                st.write(resp_txt)
                                if st.button("📋", key=f"bt_{count}", use_container_width=False):
                                    pyperclip.copy(resp_txt)
                                    st.success('Text copied successfully!')
                                if "feedback" in message:
                                    if message['feedback'] is None:
                                        feedback = streamlit_feedback(feedback_type="thumbs", key='fb_k_e', on_submit='streamlit_feedback')

            if "messages" in st.session_state and len(st.session_state.messages) > 1:
                with st.expander("Conversation History...", expanded=False):
                    with st.container(border=False):
                        if "messages" in st.session_state.keys():
                                count = -1
                                for message in st.session_state.messages:
                                    count = count + 1
                                    if count > 0:
                                        with st.chat_message(message["role"]):
                                            resp_txt = message["content"]
                                            st.write(resp_txt)
                                            if message["role"] == "assistant":
                                                if st.button("📋", key=f"bt_{count}", use_container_width=False):
                                                    pyperclip.copy(resp_txt)
                                                    st.success('Text copied successfully!')
                                                if "feedback" in message:
                                                    if message['feedback'] is None:
                                                        feedback = streamlit_feedback(feedback_type="thumbs", key=f'fb_k_{count}'
                                                                                      , on_submit='streamlit_feedback')
                                                    else:
                                                        st.write(
                                                            f"Thank you for the feedback. Your feedback was - {message['feedback']['score']}")


@st.experimental_dialog("Do you wish to continue...?")
def do_continue():
    st.write("You have selected new set of documents, this will replace the existing index of RAG!!!")
    st.write("Select 'No' to use the existing RAG index")
    if "yes_no_flag" in st.session_state:
        del st.session_state.yes_no_flag
    if st.button("Yes"):
        st.session_state.yes_no_flag = "Yes"
        st.rerun()
    if st.button("No"):
        st.session_state.yes_no_flag = "No"
        st.rerun()

def data_extractor():
    if "option" not in st.session_state:
        st.error("Please setup the application inside settings.")
    st.markdown("<b>Please upload file to setup the RAG:</b>", unsafe_allow_html=True)
    with st.form(key="Form :", clear_on_submit=True):
        doc_type = st.text_input("Document Type")
        chunking_type =  st.selectbox('Please select the chunking mechanism',('Semantic Chunking', 'Token Chunking', 'Recursive Character Text'))
        Files = st.file_uploader(label="Upload PDF file only", type=["pdf"], key='pdf',accept_multiple_files=True)
        Submit = st.form_submit_button(label='Upload & Setup RAG')

    if Submit:
        files_uploaded = []
        chunking_mechanism = chunking_type
        st.session_state.chunking_mechanism = chunking_mechanism
        if "collection_name" in st.session_state:
            del st.session_state.collection_name
        if "yes_no_flag" in st.session_state:
            del st.session_state.yes_no_flag
        save_folder = './uploaded_files'
        st.session_state.doc_type = doc_type
        for File in Files:
            st.session_state.file_upload_flag = True
            st.session_state.files_uploaded = True
            st.session_state.file_updates_found = True
            st.session_state.pdf_ref = File
            save_path = Path(save_folder, File.name)
            with open(save_path, mode='wb') as w:
                w.write(File.getvalue())
            files_uploaded.append(File.name)
        st.session_state.files_uploaded_lst = files_uploaded
        if "file_upload_flag" in st.session_state and st.session_state.file_upload_flag == True:
            st.markdown("**The files are successfully Uploaded.**")
            st.session_state.files_uploaded = files_uploaded
            model_option = None
            if "option" in st.session_state:
                model_option = st.session_state.option

            df_r = {}
            df_r["model"] = [model_option]
            df_r["doc_type"] = [doc_type.lower()]
            df_r["collection_index"] = [f"collection_DU_{doc_type.replace(" ","_").lower()}"]
            df_r["chunking_mechanism"] = [chunking_mechanism]
            df_r_pd = pd.DataFrame(df_r)
            my_file = Path("./usage_records/record_rag_index.csv")
            if my_file.is_file():
                df_tmp = pd.read_csv("./usage_records/record_rag_index.csv")
                tmp_dict_list = df_tmp.to_dict("records")
                flag_update = True
                for i in tmp_dict_list:
                    if i["doc_type"] == doc_type.lower() and i["model"] == model_option:
                        flag_update = False
                        break
                if flag_update:
                    df_r_pd.to_csv("./usage_records/record_rag_index.csv", mode ='a', index=False, header=False)
            else:
                df_r_pd.to_csv("./usage_records/record_rag_index.csv",index=False)

    if "file_upload_flag" in st.session_state and st.session_state.file_upload_flag == True:
        milvus_host = None
        milvus_port = None
        model_option = None
        llm_key = None
        if "llm_api_key" in st.session_state:
            llm_key = st.session_state.llm_api_key
        if "option" in st.session_state:
            model_option = st.session_state.option
        if "milvus_host" in st.session_state:
            milvus_host = st.session_state.milvus_host
        if "milvus_port" in st.session_state:
            milvus_port = st.session_state.milvus_port
        doc_type = st.session_state.doc_type
        chunking_mechanism = st.session_state.chunking_mechanism
        MILVUS_HOST = os.getenv('MILVUS_HOST')
        MILVUS_PORT = os.getenv('MILVUS_PORT')
        MILVUS_USERNAME = os.getenv('MILVUS_USERNAME')
        MILVUS_PASSWORD = os.getenv('MILVUS_PASSWORD')
        MILVUS_TOKEN = os.getenv('MILVUS_TOKEN')

        if MILVUS_USERNAME != "None" and MILVUS_PASSWORD != "None":
            print("With password")
            connections.connect(host=MILVUS_HOST, port=MILVUS_PORT, user=MILVUS_USERNAME, password=MILVUS_PASSWORD,
                                timeout=300)
        else:
            print("Without password")
            connections.connect(host=MILVUS_HOST, port=MILVUS_PORT,token= MILVUS_TOKEN, timeout=300)
            #connections.connect(host=MILVUS_HOST, port=MILVUS_PORT, timeout=300)

        collection_list = utility.list_collections()

        if f"collection_DU_{doc_type.replace(" ","_").lower()}" in collection_list \
                and "files_uploaded" in st.session_state:
            del st.session_state.files_uploaded
            del st.session_state.file_updates_found
            collection_name = f"collection_DU_{doc_type.replace(" ","_").lower()}"
            st.session_state.collection_name = collection_name
            do_continue()
        elif "file_updates_found" in st.session_state and f"collection_DU_{doc_type.replace(" ","_").lower()}" not in collection_list:
            st.session_state.yes_no_flag = "Yes"

        if "yes_no_flag" in st.session_state and st.session_state.yes_no_flag == "Yes":
            with st.spinner("Please wait, Rag setup in progress...."):
                if "files_uploaded_lst" in st.session_state:
                    files_uploaded = st.session_state.files_uploaded_lst
                else:
                    files_uploaded = []
                collection_name = extract_n_create_rag(files_uploaded,
                                                       doc_type.replace(" ","_").lower(),
                                                       model_option,
                                                       milvus_host,
                                                       milvus_port,
                                                       llm_key,
                                                       chunking_mechanism)
                st.session_state.collection_name = collection_name

        if "yes_no_flag" in st.session_state:
            if "collection_name" in st.session_state:
                st.success(f"RAG setup complete for document type '{doc_type}'. RAG index name is '{st.session_state.collection_name}'")



