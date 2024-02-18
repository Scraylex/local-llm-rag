import argparse
from dotenv import load_dotenv

from embedding_processor import EmbeddingProcessor

if __name__ == '__main__':
    load_dotenv()
    
    parser = argparse.ArgumentParser(description='Argument parser for input directory')
    parser.add_argument('-d', '--dir', type=str, help='Input directory path')
    args = parser.parse_args()
    input_directory = args.dir

    if not input_directory:
        raise ValueError("Input directory path is required")

    processor = EmbeddingProcessor(dir_path=input_directory)
    processor.embed_dir_to_pinecone()
