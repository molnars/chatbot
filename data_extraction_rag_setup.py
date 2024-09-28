import pandas as pd
import warnings
import os
import re
import pytesseract
from unstructured.partition.pdf import partition_pdf
import tabula
import langid
import fitz
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.text_splitter import TokenTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from pymilvus import connections, utility
from langchain_community.vectorstores import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

warnings.filterwarnings('ignore')
tesseract_cmd_path = os.getenv("TESSERACT_CMD_PATH","/usr/bin/tesseract")

def read_pdf_and_extract(file, tesseract_cmd_path):
    file_name = os.path.basename(file)
    file_name = file_name.split(".")[0]
    doc = fitz.open(file)
    page_images={}
    count = 0
    for page in doc:
        count = count + 1
        pix = page.get_pixmap(dpi=100)
        img_file_name = f"./page_images/{file_name.replace(" ","_").replace("-","_").replace(")","").replace("(","")}_{page.number}.png"
        pix.save(img_file_name)
        page_images[count] = img_file_name
    if tesseract_cmd_path is not None:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd_path
    include_page_breaks = True
    strategy =  "hi_res"
    if strategy == "hi_res":
        infer_table_structure = True
    else:
        infer_table_structure = False
    if infer_table_structure == True:
        extract_element_types=['Table']
    else:
        extract_element_types=None
    if strategy != "ocr_only":
        max_characters = None
    languages = ["eng","ara"]
    name, extension = os.path.splitext(file)
    hi_res_model_name = "yolox"
    elements = partition_pdf(
                            filename=file,
                            include_page_breaks=include_page_breaks,
                            strategy=strategy,
                            infer_table_structure=infer_table_structure,
                            extract_element_types=extract_element_types,
                            extract_images_in_pdf=False,
                            max_characters=max_characters,
                            languages=languages,
                            hi_res_model_name=hi_res_model_name,
                            )
    dict_op = {}
    for element in elements:
        page_num = element.metadata.page_number
        file_directory = element.metadata.file_directory
        file_name = element.metadata.filename
        if element.metadata.page_number not in dict_op and element.metadata.page_number is not None:
            dict_tmp = {}
            dict_tmp["page_num"] = page_num
            dict_tmp["file_directory"] = file_directory
            dict_tmp["file_name"] = file_name
            dict_tmp["page_image"] = page_images[page_num]
            txt_val = ""
            tbl_data = ""
            if "unstructured.documents.elements.Table" in str(type(element)):
                tbl_data = str(element.metadata.to_dict()["text_as_html"])
                txt_val = str(element)
            else:
                txt_val =  str(element)
                tbl_data = ""
            dict_tmp["txt_content"] = txt_val
            dict_tmp["table_content"] = txt_val
            dict_op[element.metadata.page_number] = dict_tmp
        elif element.metadata.page_number is not None:
            dict_tmp = dict_op[element.metadata.page_number]
            txt_val = ""
            tbl_data = ""
            if "unstructured.documents.elements.Table" in str(type(element)):
                tbl_data = str(element.metadata.to_dict()["text_as_html"])
                txt_val = str(element)
            else:
                txt_val =  str(element)
                tbl_data = ""
            txt_pg = dict_tmp["txt_content"]
            tbl_pg = dict_tmp["table_content"]
            dict_tmp["txt_content"] = txt_pg + "\n" + txt_val
            dict_tmp["table_content"] = tbl_pg + "\n" + tbl_data
            dict_op[element.metadata.page_number]  = dict_tmp
    return dict_op

def chunk_data_semantic(data, embedding_model):
    text_splitter = SemanticChunker( embedding_model, breakpoint_threshold_type="percentile")
    chunks = text_splitter.split_documents(data)
    return chunks

def chunk_data_token(data, chunk_size=256, chunk_overlap=20):
    text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks

def chunk_data_recursive(data, chunk_size=256, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks

def arab_to_eng_llama(txt_val):
    lang_detected = langid.classify(txt_val)[0]
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are expert in Arabic and English language.
        Output the complete text present in "Text-Data" in English language.
        NO PREAMBLE and NO EXPLANATION PLEASE.<|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        Text-Data: 
        {context} 
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        Here is the text in English:""".replace("  ", ""),
        input_variables=["context"],
    )

    llmx = ChatOllama(model="llama3.1:8b", temperature=0.1)

    # Chain
    lchain = prompt | llmx | StrOutputParser()
    if lang_detected != "en":
        docs = lchain.invoke({"context": txt_val})
        return docs, lang_detected
    else:
        return txt_val, lang_detected

def arab_to_eng_openAI(txt_val, api_key):
    lang_detected = langid.classify(txt_val)[0]
    prompt = PromptTemplate(
        template="""You are expert in Arabic and English language.
        Output the complete text present in "Text-Data" in English language.
        NO PREAMBLE and NO EXPLANATION PLEASE.
        Text-Data: 
        {context}
        
        Here is the text in English:""".replace("  ", ""),
        input_variables=["context"],
    )

    llmx = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key = api_key)
    # Chain
    lchain = prompt | llmx | StrOutputParser()

    if lang_detected != "en":
        docs = lchain.invoke({"context": txt_val})
        return docs, lang_detected
    else:
        return txt_val, lang_detected

def extract_n_create_rag(file_list, doc_type, model_option, milvus_host, milvus_port, llm_key, chunking_mechanism):
    doc_list_manual = []
    chunk_size = 512
    chunk_overlap = 10
    for file in file_list:
        print(f"Extracting details from {file}")
        x = read_pdf_and_extract("./uploaded_files/"+file, tesseract_cmd_path)
        for k, v in x.items():
            txt_val = v["txt_content"]
            tbl_data = v["table_content"]
            if model_option == "OpenAI GPT4o":
                txt_val_c, langD = arab_to_eng_openAI(txt_val, llm_key)
                table_val_c, langD_t = arab_to_eng_openAI(tbl_data, llm_key)
            else:
                txt_val_c, langD = arab_to_eng_llama(txt_val)
                table_val_c, langD_t = arab_to_eng_llama(tbl_data)
            page_num = v["page_num"]
            file_directory = v["file_directory"]
            file_name = v["file_name"]
            page_image = v["page_image"]
            data = [Document(page_content=txt_val_c, metadata={'page_num': page_num,
                                                               'file_directory': file_directory,
                                                               "source_file_name": file_name,
                                                               "page_image": page_image,
                                                               "table_content": table_val_c})]
            doc_list_manual = doc_list_manual + data

    print(f"Total number of pages -> {len(doc_list_manual)}")

    if model_option == "OpenAI GPT4o":
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=llm_key)
    else:
        OLLAMA_FLAG = str(os.getenv('OLLAMA_FLAG_EMBED', True))
        if OLLAMA_FLAG == "True":
            INFERENCE_SERVER_URL = os.getenv('EMBEDDING_SERVER_URL')
            model_name_embd = os.getenv('EMBEDDING_MODEL_NAME','nomic-embed-text')
            embedding_model = OllamaEmbeddings(model=model_name_embd,  base_url=INFERENCE_SERVER_URL)
        else:
            model_kwargs = {'trust_remote_code': True}
            embedding_model = HuggingFaceEmbeddings(
                                                    model_name="nomic-ai/nomic-embed-text-v1",
                                                    model_kwargs=model_kwargs,
                                                    show_progress=False
                                                )

    if chunking_mechanism == 'Semantic Chunking':
        dt = chunk_data_semantic(doc_list_manual, embedding_model)
    elif chunking_mechanism == 'Token Chunking':
        dt = chunk_data_token(doc_list_manual, chunk_size, chunk_overlap)
    elif chunking_mechanism == 'Recursive Character Text':
        dt = chunk_data_recursive(doc_list_manual, chunk_size, chunk_overlap)

    MILVUS_HOST = os.getenv('MILVUS_HOST')
    MILVUS_PORT = os.getenv('MILVUS_PORT')
    MILVUS_USERNAME = os.getenv('MILVUS_USERNAME')
    MILVUS_PASSWORD = os.getenv('MILVUS_PASSWORD')
    MILVUS_TOKEN = os.getenv('MILVUS_TOKEN')

    if MILVUS_USERNAME != "None" and MILVUS_PASSWORD != "None":
        print("With password")
        connection_args_dict = {"host": MILVUS_HOST, "port": MILVUS_PORT, "user": MILVUS_USERNAME,
                                "password": MILVUS_PASSWORD, "timeout": 300}
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT, user=MILVUS_USERNAME, password=MILVUS_PASSWORD, timeout=300)

    else:
        print("Without password")
        connection_args_dict = {"host": MILVUS_HOST, "port": MILVUS_PORT, "token": MILVUS_TOKEN, "timeout": 300}
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT, token=MILVUS_TOKEN, timeout=300)
        #connection_args_dict = {"host": MILVUS_HOST, "port": MILVUS_PORT, "timeout": 300}
        #connections.connect(host=MILVUS_HOST, port=MILVUS_PORT, timeout=300)

    collection_list = utility.list_collections()
    if f"collection_DU_{doc_type}" in collection_list:
        utility.drop_collection(f"collection_DU_{doc_type}")
    print("Creating RAG index")
    vectorstore_collection = Milvus.from_documents(documents=dt,
                                               embedding=embedding_model,
                                               collection_name=f"collection_DU_{doc_type}",
                                               connection_args=connection_args_dict)
    return f"collection_DU_{doc_type}"