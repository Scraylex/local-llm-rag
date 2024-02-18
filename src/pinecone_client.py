import os
import time

from dotenv import load_dotenv
from pinecone import Pinecone, Index, PodSpec
from util import create_embedding_model


class PineconeClient:
    _max_retries = 5

    def __init__(self):
        load_dotenv()
        api_key = os.getenv("PINECONE_API_KEY")
        self.env = os.getenv("PINECONE_ENV")
        self.idx_name = os.getenv("PINECONE_IDX")
        self.pc = Pinecone(api_key=api_key, environment=self.env)

    def ready_check(self) -> None:
        model_name = os.getenv('EMBEDDING_MODEL_NAME')

        if model_name is None:
            raise ValueError("Please set the EMBEDDING_MODEL_NAME environment variable")

        docs = [
            "this is one document",
        ]

        embeddings = create_embedding_model(model_name=model_name, batch_size=1).embed_documents(docs)

        if self.idx_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                self.idx_name,
                dimension=len(embeddings[0]),
                metric='cosine',
                spec=PodSpec(environment=self.env)
            )
            while not self.pc.describe_index(self.idx_name).status['ready']:
                time.sleep(1)

    def connect(self) -> Index:
        return self.pc.Index(self.idx_name)

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
