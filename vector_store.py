from src.helper import load_pdf_file, text_split, download_google_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
google_api_key = os.environ.get("GOOGLE_API_KEY")

os.environ["PINECONE_API_KEY"] = pinecone_api_key
os.environ["GOOGLE_API_KEY"] = google_api_key

extracted_data = load_pdf_file(data = "data/")
text_chunks = text_split(extracted_data)
embeddings = download_google_embeddings()

pc = Pinecone(api_key = pinecone_api_key)

index_name = "fitbot-index"

pc.create_index(
    name = index_name,
    dimension = 768,
    metric = "cosine",
    spec = ServerlessSpec(
        cloud = "aws",
        region = "us-east-1"
    )
)

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings, 
)