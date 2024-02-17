import os
import time

import pinecone
from dotenv import load_dotenv
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from langchain.vectorstores.pinecone import Pinecone as VectorStorePinecone


class PineconeClient:
    _max_retries = 5

    def __init__(self):
        load_dotenv()
        api_key = os.getenv("PINECONE_API_KEY")
        self.env = os.getenv("PINECONE_ENV")
        self.idx_name = os.getenv("PINECONE_IDX")
        self.pc = Pinecone(api_key=api_key, environment=self.env)

    def ready_check(self, embed_model: HuggingFaceEmbeddings) -> None:
        docs = [
            "this is one document",
        ]
        embeddings = embed_model.embed_documents(docs)

        if self.idx_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                self.idx_name,
                dimension=len(embeddings[0]),
                metric='cosine',
                spec=pinecone.PodSpec(environment=self.env)
            )
            while not self.pc.describe_index(self.idx_name).status['ready']:
                time.sleep(1)

    def connect(self) -> pinecone.Index:
        return self.pc.Index(self.idx_name)

    def get_as_vectorstore(self) -> VectorStorePinecone:
        index = self.connect()
        text_field = 'text'  # field in metadata that contains text content
        vectorstore = VectorStorePinecone(index, text_field=text_field)
        return vectorstore

    def upsert_to_pinecone(self, file_path: str, to_upsert) -> None:
        index = self.connect()
        try:
            index.upsert(vectors=to_upsert)
        except Exception as _:
            print(f"Transmit failed")
            done = False
            counter = 0
            while not done and counter < self._max_retries:
                print("retransmitting")
                time.sleep(1)
                try:
                    index.upsert(vectors=to_upsert)
                    done = True
                except Exception as _:
                    print(f"Transmit failed again")
                    counter += 1
                    pass
            if not done:
                print(f"Failed to transmit {file_path}")
