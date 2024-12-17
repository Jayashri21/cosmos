from http.client import HTTPException
from azure.cosmos import CosmosClient, PartitionKey, exceptions
import json
import os
from pathlib import Path
from typing import List, Optional, Union
import sys
from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBED_MODEL,
    COSMOS_CONNECTION_STRING,
    COSMOS_DB_NAME,
    COSMOS_DB_CONTAINER,
    LOG_FLAG,
    TEI_ENDPOINT,
    UPLOADED_FILES_PATH,
    HUGGINGFACE_TOKEN,
    endpoint,
    key
)
from langchain.text_splitter import HTMLHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from comps import CustomLogger, DocPath, opea_microservices, register_microservice

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from comps.dataprep.utils import (
    create_upload_folder,
    document_loader,
    encode_filename,
    get_file_structure,
    get_separators,
    get_tables_result,
    parse_html,
    remove_folder_with_ignore,
    save_content_to_local_disk,
    
)
from fastapi import FastAPI, UploadFile, File, HTTPException, Form


logger = CustomLogger(__name__)

# Cosmos DB Client setup
cosmos_client = CosmosClient(endpoint, key)
database = cosmos_client.get_database_client(COSMOS_DB_NAME)
container = database.get_container_client(COSMOS_DB_CONTAINER)


logger = CustomLogger(__name__)

app = FastAPI()

model_name = "sentence-transformers/all-MiniLM-L6-v2"

def get_embedder() -> Union[HuggingFaceEndpointEmbeddings, HuggingFaceBgeEmbeddings]:
    """Obtain required Embedder."""
    if TEI_ENDPOINT:
        return HuggingFaceEndpointEmbeddings(model=TEI_ENDPOINT, hf_token=HUGGINGFACE_TOKEN)
    else:
        return HuggingFaceBgeEmbeddings(model_name=EMBED_MODEL, hf_token=HUGGINGFACE_TOKEN)

def search_by_filename(file_name: str) -> bool:
    """Search Cosmos DB by file name."""
    query = f"SELECT * FROM c WHERE c.metadata.doc_name = '{file_name}'"
    results = list(container.query_items(query=query, enable_cross_partition_query=True))

    if LOG_FLAG:
        logger.info(f"[search by file] searched by {file_name}")
        logger.info(f"[search by file] {len(results)} results: {results}")

    return len(results) > 0





def ingest_doc_to_cosmos(doc_path: DocPath) -> None:
    """Ingest documents to Cosmos DB."""
    
    path = doc_path.path
    file_name = path.split("/")[-1]
    if LOG_FLAG:
        logger.info(f"Parsing document {path}, file name: {file_name}.")

    if path.endswith(".html"):
        headers_to_split_on = [
            ("h1", "Header 1"),
            ("h2", "Header 2"),
            ("h3", "Header 3"),
        ]
        text_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    else:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=doc_path.chunk_size,
            chunk_overlap=doc_path.chunk_overlap,
            add_start_index=True,
            separators=get_separators(),
        )

    content = document_loader(path)

    structured_types = [".xlsx", ".csv", ".json", "jsonl"]
    _, ext = os.path.splitext(path)

    if ext in structured_types:
        chunks = content
    else:
        chunks = text_splitter.split_text(content)

    if doc_path.process_table and path.endswith(".pdf"):
        table_chunks = get_tables_result(path, doc_path.table_strategy)
        chunks = chunks + table_chunks

    if LOG_FLAG:
        logger.info(f"Done preprocessing. Created {len(chunks)} chunks of the original file.")

    if len(chunks) == 0:
        logger.error(f"No chunks created for document {file_name}. Exiting.")
        return  # If no chunks are created, exit the function.

    batch_size = 32
    num_chunks = len(chunks)

    metadata = dict({"doc_name": str(file_name)})

    for i in range(0, num_chunks, batch_size):
        batch_chunks = chunks[i: i + batch_size]
        batch_texts = batch_chunks

        # Create documents for Cosmos DB ingestion
        documents = [{"id": f"{file_name}-{i}", "content": text, "metadata": metadata} for i, text in enumerate(batch_texts)]

        for doc in documents:
            try:
                # Insert document into Cosmos DB container
                container.upsert_item(doc)
                if LOG_FLAG:
                    logger.info(f"Processed batch {i // batch_size + 1}/{(num_chunks - 1) // batch_size + 1}")
            except exceptions.CosmosResourceExistsError:
                logger.error(f"Document with id {doc['id']} already exists in Cosmos DB.")
            except Exception as e:
                logger.error(f"Error ingesting document {doc['id']} to Cosmos DB: {str(e)}")









@app.post("/ingest_documents")
async def ingest_documents(
    files: Optional[Union[UploadFile, List[UploadFile]]] = File(None),
    link_list: Optional[str] = Form(None),
    chunk_size: int = Form(1500),
    chunk_overlap: int = Form(100),
    process_table: bool = Form(False),
    table_strategy: str = Form("fast"),
):
    """Ingest documents for RAG."""
    
    if LOG_FLAG:
        logger.info(f"files:{files}")
        logger.info(f"link_list:{link_list}")

    if files and link_list:
        raise HTTPException(status_code=400, detail="Provide either a file or a string list, not both.")

    if files:
        if not isinstance(files, list):
            files = [files]

        if not os.path.exists(UPLOADED_FILES_PATH):
            Path(UPLOADED_FILES_PATH).mkdir(parents=True, exist_ok=True)

        for file in files:
            encode_file = encode_filename(file.filename)
            save_path = UPLOADED_FILES_PATH + encode_file
            filename = save_path.split("/")[-1]

            exists = search_by_filename(filename)
            if exists:
                if LOG_FLAG:
                    logger.info(f"[upload] File {file.filename} already exists.")
                raise HTTPException(
                    status_code=400,
                    detail=f"Uploaded file {file.filename} already exists. Please change the file name.",
                )

            await save_content_to_local_disk(save_path, file)

            # Log the document ingestion process
            if LOG_FLAG:
                logger.info(f"Successfully saved file {save_path}.")

            # Ingest to Cosmos DB
            ingest_doc_to_cosmos(
                DocPath(
                    path=save_path,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    process_table=process_table,
                    table_strategy=table_strategy,
                )
            )

        result = {"status": 200, "message": "Data preparation succeeded"}

        if LOG_FLAG:
            logger.info(result)
        return result
    

    
    raise HTTPException(status_code=400, detail="Must provide either a file or a string list.")


# Helper function to encode the filename (you can modify this)
def encode_filename(filename: str) -> str:
    """Encode the filename by replacing spaces and converting it to lowercase."""
    return filename.replace(" ", "_").lower()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
