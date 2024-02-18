import openai
import os
from dotenv import load_dotenv

from util import create_embedding_model
from pinecone_client import PineconeClient


class RagClient:

    def __init__(self):
        load_dotenv()
        model_name = os.getenv("EMBEDDING_MODEL_NAME")
        base_url = os.getenv("LOCAL_LLM_BASE_URL")
        api_key = 'rndm-key'
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.pc_client = PineconeClient()
        self.embed_model = create_embedding_model(model_name=model_name, batch_size=1)

    def query_rag(self, query: str):
        res = self.embed_model.embed_query(query)
        docs = self.pc_client.connect().query(vector=res, top_k=3, include_metadata=True)['matches']
        contexts = [item['metadata']['text'] for item in docs]
        augmented_query = "\n\n---\n\n".join(contexts) + "\n\n-----\n\n"

        full_query = augmented_query + query

        # system message to 'prime' the model
        primer = """You are a bot specialized in Life Cycle Analysis data retrieval.
        To achieve this goal JSON resources will be included in the user's query as text.
        Your task is to analyze the provided JSON resources and summarize their content.
        For each individual JSON record always list all records. Always extract the values of the
        'activity' and 'name' keys from the JSONs and list them in the response.
        """

        res = self.client.chat.completions.create(
            model="local-model",
            messages=[
                {"role": "system", "content": primer},
                {"role": "user", "content": full_query}
            ]
        )
        return res.choices[0].message.content
