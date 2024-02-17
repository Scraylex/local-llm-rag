import argparse
from dotenv import load_dotenv

from embedding_processor import EmbeddingProcessor

if __name__ == '__main__':
    load_dotenv()
    
    parser = argparse.ArgumentParser(description='Argument parser for input directory')
    parser.add_argument('-d', '--dir', type=str, help='Input directory path')
    parser.add_argument('-m', '--model', type=str, default='all-MiniLM-L6-v2',
                        help='Model name for tokenization and embedding')
    parser.add_argument('-b', '--batch', type=int, default=32, help='Batch size for embedding')

    args = parser.parse_args()
    input_directory = args.dir
    model_name = args.model
    batch_size = args.batch

    if not input_directory:
        raise ValueError("Input directory path is required")

    processor = EmbeddingProcessor(dir_path=input_directory, model=model_name, batch=batch_size)
    processor.embed_dir_to_pinecone()
