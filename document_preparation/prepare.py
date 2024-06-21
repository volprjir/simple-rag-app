import os

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from config import EMBEDDING_MODEL

DB_FAISS_PATH = os.path.join("document_preparation", "db_faiss")
OPEN_API_KEY = os.getenv("OPENAI_API_KEY")
if OPEN_API_KEY is None:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

if __name__ == "__main__":
    try:
        with open(
            os.path.join("document_preparation", "no_sale_countries.md"), "r"
        ) as file:
            lines = file.readlines()
            splits = [line.strip() for line in lines]
            faiss_processing = FAISS.from_texts(
                texts=splits,
                embedding=OpenAIEmbeddings(api_key=OPEN_API_KEY, model=EMBEDDING_MODEL),
            )
            faiss_processing.save_local(DB_FAISS_PATH)
            print("DB FAISS saved successfully.")
    except Exception as e:
        print(f"Error: {e}")
