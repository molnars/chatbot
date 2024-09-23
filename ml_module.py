from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.schema.document import Document
import pandas as pd
import warnings
import os
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel,Field
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain.output_parsers import OutputFixingParser
from langchain_community.llms import VLLMOpenAI
warnings.filterwarnings('ignore')


class src_dtls(BaseModel):
    page_number: list = Field(description="List of page numbers from context from which the query is mostly answered")
    file_name : list = Field(description="List of file names from context from which the query is mostly answered")
    page_image: list = Field(description="List of page_image paths from which the query is mostly answered")
class query_answer(BaseModel):
    answer: str = Field(description="Answer to the query based on the context in string format or text format")
    source_details : src_dtls = Field(description="Source details like list of page_number, file_name and page_image.")

parser = JsonOutputParser(pydantic_object=query_answer)

def answer_query_openai(api_key, query, context_doc, rag_prompt):
    prompt_template_GPT4oq = """Your expert in answering the query based on only the context provided.
    There might be few tables in the context provided, in HTML format for your better understanding.
    
    Context: 
    {context}

    Query :{query}
    
    {rag_prompt}
    
    ONLY INCLUDE DETAILS IN YOUR ANSWER WHICH IS ASKED IN THE 'Query'.
    Also extract and mention all the page_number, file_name and page_image from which the query is mostly answered, from the Context.
    If the context is irrelevant to the query, return ["NA"] for each key of source_details.
    "answer" key should be in Text format.
     
    {format_instructions}
    
    No preamble please.
    """
    MAX_TOKENS = int(os.getenv('MAX_TOKENS', 1024))
    TEMPERATURE = float(os.getenv('TEMPERATURE', 0.1))
    llm_model = ChatOpenAI(model="gpt-4o",
                           temperature=TEMPERATURE,
                           openai_api_key=api_key,
                           max_tokens=MAX_TOKENS)

    prompt_GPT4oq = PromptTemplate(
        template=prompt_template_GPT4oq,
        input_variables=["query", "context", "rag_prompt"],
        partial_variables = {"format_instructions": parser.get_format_instructions()}
    )
    new_parser = OutputFixingParser.from_llm(parser=parser, llm=llm_model, max_retries=3)

    chain_GPT4oq = prompt_GPT4oq | llm_model | new_parser
    print("Here inside answer_query_openai")
    return chain_GPT4oq.invoke({"query": query, "context": "\n\n".join(context_doc),"rag_prompt":rag_prompt})


def answer_general_query_openai(api_key, query, general_prompt):
    prompt_template_GPT4oq = """Your expert in answering the query based on your own knowledge.
    
    Query :{query}

    {general_prompt}
    ONLY INCLUDE DETAILS IN YOUR ANSWER WHICH IS ASKED IN THE 'Query'.
    No preamble please.
    """

    MAX_TOKENS = int(os.getenv('MAX_TOKENS', 1024))
    TEMPERATURE = float(os.getenv('TEMPERATURE', 0.1))

    llm_model = ChatOpenAI(model="gpt-4o",
                           temperature=TEMPERATURE,
                           openai_api_key=api_key,
                           max_tokens=MAX_TOKENS)

    prompt_GPT4oq = PromptTemplate(
                                    template=prompt_template_GPT4oq,
                                    input_variables=["query","general_prompt"]
                                )

    chain_GPT4oq = prompt_GPT4oq | llm_model | StrOutputParser()
    print("Here inside answer_general_query_openai")
    return chain_GPT4oq.invoke({"query": query,"general_prompt":general_prompt})


def answer_query_llama(query, context_doc, rag_prompt):
    prompt_template_llama = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    Your expert in answering the query in detail, based on only the context provided.
    There might be few tables in the context provided, in HTML format for your better understanding.
    <|eot_id|>

    <|start_header_id|>user<|end_header_id|>
    Context: 
    {context}

    Query :{query}

    {rag_prompt}
    ONLY INCLUDE DETAILS IN YOUR ANSWER WHICH IS ASKED IN THE 'Query'.

    Also extract and mention all the page_number, file_name and page_image from which the query is mostly answered, from the Context.
    If the Context is irrelevant to the Query, return ["NA"] for each key of source_details.

    Your JSON Output should include the following keys and format:
    {{"answer" : <Your answer to the query in text format>,
    "source_details": {{"page_number" : <Array of page numbers or NA>,
                        "file_name": <Array of file names or NA>,
                        "page_image": <Array of page images or NA>}}
    }}
    
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    Here is the Answer to the Query in JSON format:
    """

    MAX_TOKENS = int(os.getenv('MAX_TOKENS', 4000))
    TEMPERATURE = float(os.getenv('TEMPERATURE', 0.1))

    OLLAMA_FLAG = str(os.getenv('OLLAMA_FLAG', True))
    if OLLAMA_FLAG == "True":
        INFERENCE_SERVER_URL = os.getenv('LLM_INFERENCE_SERVER_URL')
        MODEL_NAME = os.getenv('MODEL_NAME', "llama3.1:8b")
        llmx = ChatOllama(model=MODEL_NAME, temperature=TEMPERATURE, num_predict=MAX_TOKENS,
                          base_url=INFERENCE_SERVER_URL)
        llmx_p = ChatOllama(model=MODEL_NAME, temperature=0.0,
                            num_predict=1000,
                            base_url=INFERENCE_SERVER_URL)
    else:
        INFERENCE_SERVER_URL = os.getenv('LLM_INFERENCE_SERVER_URL')
        MODEL_NAME = os.getenv('MODEL_NAME', "llama3.1:8b")
        llmx = VLLMOpenAI(
            openai_api_key="EMPTY",
            openai_api_base=INFERENCE_SERVER_URL,
            model_name=MODEL_NAME,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE
        )
        llmx_p = VLLMOpenAI(
            openai_api_key="EMPTY",
            openai_api_base=INFERENCE_SERVER_URL,
            model_name=MODEL_NAME,
            max_tokens=MAX_TOKENS,
            temperature=0.0
        )

    prompt_llama = PromptTemplate(
        template=prompt_template_llama,
        input_variables=["query", "context", "rag_prompt"],
        # partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    new_parser = OutputFixingParser.from_llm(parser=parser, llm=llmx_p, max_retries=5)
    chain_llama = prompt_llama | llmx | StrOutputParser()
    # chain_llama = prompt_llama | llmx | new_parser
    print("Here inside answer_query_llama")
    op_from_chain = chain_llama.invoke({"query": query, "context": "\n\n".join(context_doc), "rag_prompt": rag_prompt})
    print(op_from_chain.replace("**",""))
    op_final = new_parser.parse(op_from_chain.replace("**",""))

    print("Done")
    print(op_final)
    return op_final


def answer_general_query_llama(query,general_prompt):
    prompt_template_llama = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    Your expert in answering the query in detail, based on your own knowledge
    <|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    Query :{query}

    {general_prompt}
    ONLY INCLUDE DETAILS IN YOUR ANSWER WHICH IS ASKED IN THE 'Query'.
    
    No preamble please.
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    Here is the Answer:
    """
    MAX_TOKENS = int(os.getenv('MAX_TOKENS', 1024))
    TEMPERATURE = float(os.getenv('TEMPERATURE', 0.1))

    OLLAMA_FLAG = str(os.getenv('OLLAMA_FLAG', True))
    if OLLAMA_FLAG == "True":
        INFERENCE_SERVER_URL = os.getenv('LLM_INFERENCE_SERVER_URL')
        MODEL_NAME = os.getenv('MODEL_NAME')
        llmx = ChatOllama(model=MODEL_NAME, temperature=TEMPERATURE, num_predict=MAX_TOKENS,
                          base_url=INFERENCE_SERVER_URL)
    else:
        INFERENCE_SERVER_URL = os.getenv('LLM_INFERENCE_SERVER_URL')
        MODEL_NAME = os.getenv('MODEL_NAME')
        llmx = VLLMOpenAI(
                            openai_api_key="EMPTY",
                            openai_api_base=INFERENCE_SERVER_URL,
                            model_name=MODEL_NAME,
                            max_tokens=MAX_TOKENS,
                            temperature=TEMPERATURE
                        )

    prompt_llama = PromptTemplate(
        template=prompt_template_llama,
        input_variables=["query","general_prompt"]
    )
    chain_llama = prompt_llama | llmx | StrOutputParser()
    print("Here inside answer_general_query_llama")
    op_from_chain = chain_llama.invoke({"query": query,"general_prompt":general_prompt})

    print("Done")
    return op_from_chain