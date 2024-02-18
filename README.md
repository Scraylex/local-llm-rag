# local-llm-rag

This is a local llm rag version of a small streamlit frontend.

## Installation

```bash
pip install -r requirements.txt
```

create a .env file with the following keys set

```bash
PINECONE_API_KEY=<API-KEY>
PINECONE_ENV=gcp-starter # this is default for free tier of pinecone
PINECONE_IDX=local-rag # some name for the index
LOCAL_LLM_BASE_URL=http://localhost:1234/v1 # this has to match the url of the local inference server using lmstudio.ai
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2 # one of the sentence transformer models available from huggingface
BATCH_SIZE=32 # batch size for the embedding model
```

## Usage

To embed the files present in a directory (currently .pdf supported)

```bash
python embedding.py -d "/path/to/directory"
```

To run the streamlit frontend chatbot

```bash
streamlit run app.py
```


