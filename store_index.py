from src.helper import  load_pdf_file ,text_split ,download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

extracted_data=load_pdf_file(data='data/')
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()



# Make sure the API key is fetched correctly
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")  # Ensure this environment variable is set

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medicalbot1"

# Create the index
pc.create_index(
    name=index_name,
    dimension=384,  # This must match the output dimension of your embedding model
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",        # You can also use "gcp" if needed
        region="us-east-1"  # Must match your Pinecone project setup
    )
)


docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings, 
)