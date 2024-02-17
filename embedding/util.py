from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from torch import cuda
import PyPDF2


def create_text_splitter(max_seq_len: int, overlap=30) -> RecursiveCharacterTextSplitter:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_seq_len,
        chunk_overlap=overlap
    )
    return text_splitter


def create_embedding_model(model_name: str = 'all-MiniLM-L6-v2', batch_size: int = 32) -> HuggingFaceEmbeddings:
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
    model_kwargs = {'device': device}
    encode_kwargs = {
        'normalize_embeddings': False,  # else normalize for dot product pinecone index
        'batch_size': batch_size,
        'device': device
    }
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,  # Provide the pre-trained model's path
        model_kwargs=model_kwargs,  # Pass the model configuration options
        encode_kwargs=encode_kwargs  # Pass the encoding options
    )
    return embeddings


def extract_text_from_pdf(pdf_path: str) -> str:
    # Open the PDF file
    with open(pdf_path, 'rb') as file:
        # Create PDF reader object
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''

        # Iterate through each page
        for page_num in range(len(pdf_reader.pages)):
            # Get a page by index
            page = pdf_reader.pages[page_num]

            # Extract text from the page
            text += page.extract_text() + "\n"

    return text
