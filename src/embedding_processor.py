import os

from tqdm import tqdm

from util import create_text_splitter, create_embedding_model, extract_text_from_pdf
from pinecone_client import PineconeClient


class EmbeddingProcessor:
    def __init__(self, dir_path: str):
        batch = int(os.getenv('BATCH_SIZE', '32'))
        model = os.getenv('MODEL_NAME', 'all-MiniLM-L6-v2')
        self.dir_path = dir_path
        self.pc_client = PineconeClient()
        self.embed_model = create_embedding_model(batch_size=batch, model_name=model)
        self.text_splitter = create_text_splitter(self.embed_model.client.tokenizer.model_max_length)

    def embed_dir_to_pinecone(self) -> None:
        self.pc_client.ready_check()
        if os.path.exists(self.dir_path):
            sorted_filenames = sorted(os.listdir(self.dir_path))
            for filename in tqdm(sorted_filenames):
                self._process_file(filename)
        print(self.pc_client.connect().describe_index_stats())

    def _process_file(self, filename: str) -> None:
        file_path = os.path.join(self.dir_path, filename)
        batch_size = self.embed_model.encode_kwargs['batch_size']
        chunks = []
        file_text = extract_text_from_pdf(file_path)
        texts = self.text_splitter.split_text(file_text)
        chunks.extend([{
            'id': f'{filename}-{i}',
            'text': texts[i],
            'chunk': i
        } for i in range(len(texts))])
        for i in range(0, len(chunks), batch_size):
            # find end of batch
            i_end = min(len(chunks), i + batch_size)
            meta_batch = chunks[i:i_end]
            # get ids
            ids_batch = [x['id'] for x in meta_batch]
            # get texts to encode
            texts = [x['text'] for x in meta_batch]
            embeds = self.embed_model.embed_documents(texts)
            meta_batch = [{
                'text': x['text'],
                'chunk': x['chunk'],
                'id': x['id']
            } for x in chunks]
            to_upsert = list(zip(ids_batch, embeds, meta_batch))
            # upsert to Pinecone
            self.pc_client.upsert_to_pinecone(file_path, to_upsert)
