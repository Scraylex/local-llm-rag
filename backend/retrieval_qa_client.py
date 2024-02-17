import os
from typing import Any, Dict

import openai
from langchain.chains import RetrievalQA
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI

from web.pinecone_client import PineconeClient


def _init_openai():
    some_key = "sx-xxx"  # use a dummy key
    openai.api_base = "http://localhost:8080/"  # specify where the server runs
    openai.api_key = some_key
    os.environ['OPENAI_API_KEY'] = some_key  # set the environment variable


class RetrievalQAClient:

    def __init__(self):
        _init_openai()
        self.pc_client = PineconeClient()
        self.retrievalQA = self._create_retrieval_pipeline()
        
    def do_rag(self, query) -> Dict[str, Any]:
        return self.retrievalQA.invoke(query)

    def _create_retrieval_pipeline(self) -> RetrievalQA:
        vectorstore = self.pc_client.get_as_vectorstore()
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        
        template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer. 
        Use three sentences maximum and keep the answer as concise as possible. 
        Always say "thanks for asking!" at the end of the answer. 
        {context}
        Question: {question}
        Helpful Answer:"""

        qa_prompt = PromptTemplate(input_variables=["context", "question"], template=template)

        return RetrievalQA.from_chain_type(
            llm=llm, chain_type='stuff',
            retriever=vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": qa_prompt}
        )