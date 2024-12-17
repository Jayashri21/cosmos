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
        return HuggingFaceBgeEmbeddings(model_name=EMBED_MODEL)

def generate_vector(content: str):
    """Generate vector embeddings for the given content."""
    try:
        # Obtain the embedder object
        embedder = get_embedder()  # Call get_embedder() to get the embedder instance
        # Get embeddings from the chosen embedder
        embeddings = embedder.embed_documents([content])   # Pass content to embedder's embed method
        return embeddings
    except Exception as e:
        logger.error(f"Error generating vector for content: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating vector for content: {str(e)}")

def ingest_doc_to_cosmos(doc_path: DocPath, file) -> None:
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

        # Create documents for Cosmos DB ingestion, adding the vector field
        documents = []
        for i, text in enumerate(batch_texts):
            vector = generate_vector(text)  # Generate vector for each chunk
            doc = {
                "id": f"{file_name}-{i}",
                "content": text,
                "metadata": metadata,
                "file_name": file.filename,
                "vector": vector  # Add vector field
            }
            documents.append(doc)

        # Ingest documents in batches to Cosmos DB
        for doc in documents:
            try:
                container.upsert_item(doc)
                if LOG_FLAG:
                    logger.info(f"Processed batch {i // batch_size + 1}/{(num_chunks - 1) // batch_size + 1}")
            except exceptions.CosmosResourceExistsError:
                logger.error(f"Document with id {doc['id']} already exists in Cosmos DB.")
            except Exception as e:
                logger.error(f"Error ingesting document {doc['id']} to Cosmos DB: {str(e)}")


# Ingest links with vector embeddings
async def ingest_link_to_cosmos(link_list: List[str]) -> None:
    """Ingest data scraped from website links into Cosmos DB."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
        separators=get_separators(),
    )

    batch_size = 32

    for link in link_list:
        try:
            print(link , "Link")
            # Sanitize the link to create a valid file name
            encoded_link = sanitize_filename(link)
            save_path = os.path.join(UPLOADED_FILES_PATH, f"{encoded_link}.txt")
            content = parse_html([link])[0][0]
            if LOG_FLAG:
                logger.info(f"[ ingest link ] link: {link} content: {content}")

            await save_content_to_local_disk(save_path, content)

            chunks = text_splitter.split_text(content)
            num_chunks = len(chunks)
            metadata = dict({"doc_name": str(save_path)})

            # Ingest each chunk with vector embedding
            for i in range(0, num_chunks, batch_size):
                batch_chunks = chunks[i: i + batch_size]
                batch_texts = batch_chunks

                documents = []
                for i, text in enumerate(batch_texts):
                    vector = generate_vector(text)  # Generate vector for each chunk
                    doc = {
                        "id": f"{encoded_link}-{i}",
                        "content": text,
                        "metadata": metadata,
                        "file_name": link,
                        "vector": vector  # Add vector field
                    }
                    documents.append(doc)

                # Ingest documents in batches to Cosmos DB
                for doc in documents:
                    try:
                        container.upsert_item(doc)
                        if LOG_FLAG:
                            logger.info(f"Processed batch {i // batch_size + 1}/{(num_chunks - 1) // batch_size + 1}")
                    except exceptions.CosmosResourceExistsError:
                        logger.error(f"Document with id {doc['id']} already exists in Cosmos DB.")
                    except Exception as e:
                        logger.error(f"Error ingesting document {doc['id']} to Cosmos DB: {str(e)}")
        except Exception as e:
            logger.error(f"Error occurred while processing the link {link}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing link {link}: {str(e)}")

def search_by_filename(file_name: str) -> bool:
    """Search Cosmos DB by file name."""
    query = f"SELECT * FROM c WHERE c.metadata.doc_name = '{file_name}'"
    results = list(container.query_items(query=query, enable_cross_partition_query=True))

    if LOG_FLAG:
        logger.info(f"[search by file] searched by {file_name}")
        logger.info(f"[search by file] {len(results)} results: {results}")

    return len(results) > 0


import re

def sanitize_filename(url: str) -> str:
    """Sanitize the URL to make it a valid filename."""
    # Replace invalid characters with underscores
    return re.sub(r'[:/\\?%*|"<>\.]', '_', url)


@register_microservice(
    name="opea_service@prepare_doc_elastic",
    endpoint="/v1/dataprep/get_file",
    host="0.0.0.0",
    port=6011,
)
async def rag_get_file_structure_from_cosmos() -> list:
    """Obtain the list of documents stored in Cosmos DB."""

    if LOG_FLAG:
        logger.info("[ dataprep - get file structure ] start to get file structure from Cosmos DB")

    # Query Cosmos DB for all documents (or filter by a certain field if needed)
    query = "SELECT c.metadata.doc_name FROM c"

    try:
        # Execute the query to get the file structure (list of document names)
        results = list(container.query_items(query=query, enable_cross_partition_query=True))
        
        if results:
            # Extract the document names from the query results
            file_content = list(dict.fromkeys(result["doc_name"] for result in results))


            if LOG_FLAG:
                logger.info(f"Retrieved file structure from Cosmos DB: {file_content}")
            
            return file_content
        else:
            if LOG_FLAG:
                logger.info("No files found in Cosmos DB.")
            return []

    except exceptions.CosmosHttpResponseError as e:
        logger.error(f"Error occurred while querying Cosmos DB: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving file structure from Cosmos DB.")


@register_microservice(name="opea_service@prepare_doc_elastic", endpoint="/v1/dataprep", host="0.0.0.0", port=6011)
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
        if isinstance(files, UploadFile):  # Single file
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
                ) , file
            )

        result = {"status": 200, "message": "Data preparation succeeded"}

        if LOG_FLAG:
            logger.info(result)
        return result
    

    if link_list:
        try:
            print("link_list before parsing:", link_list)
            link_list = json.loads(link_list)  # Parse the string into a list
            print("link_list after parsing:", link_list)
            if not isinstance(link_list, list):
                raise HTTPException(status_code=400, detail="link_list must be a list.")
            if not all(isinstance(link, str) for link in link_list):
                raise HTTPException(status_code=400, detail="Each item in link_list must be a string.")
            
            await ingest_link_to_cosmos(link_list)

            if LOG_FLAG:
                logger.info(f"Successfully saved link list {link_list}")

            result = {"status": 200, "message": "Data preparation succeeded"}
            return result

        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format for link_list.")
        except Exception as e:
            print(f"Exception occurred: {str(e)}")  # Log the exception for debugging
            raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@register_microservice(
    name="opea_service@prepare_doc_elastic",
    endpoint="/v1/dataprep/delete_file",
    host="0.0.0.0",
    port=6011,
)
async def delete_file(file_path: str):
    """Delete a file or all files from Cosmos DB."""
    result = await delete_single_file(file_path)
    if result["status"]:
        return result
    else:
        raise HTTPException(status_code=404, detail=result["message"])

async def delete_single_file(file_path: str):
    """Delete file according to `file_path` in Cosmos DB.
    
    `file_path`:
        - specific file ID (e.g. laxmi_shrinivas_gajjam.txt-0)
        - "all": delete all documents in Cosmos DB
    """
    if file_path == "all":
        if LOG_FLAG:
            logger.info("[dataprep - del] delete all files from Cosmos DB")
        try:
            # Query all items in the container
            for item in container.read_all_items():
                file_id = item['file_name']
                partition_key = item['file_name']  # Use 'file_name' as the partition key
                # Delete each document
                delete_result = delete_document_from_cosmos(file_id, partition_key)
                if LOG_FLAG:
                    logger.info(f"Successfully deleted file: {file_id}")
            if LOG_FLAG:
                logger.info("[dataprep - del] successfully deleted all files from Cosmos DB.")
            return {"status": True}

        except Exception as e:
            if LOG_FLAG:
                logger.error(f"[dataprep - del] failed to delete all files: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to delete all files: {str(e)}")

    # Delete a specific file (document)
    delete_path = file_path  # Assuming `file_path` is the document ID

    if LOG_FLAG:
        logger.info(f"[dataprep - del] delete document with ID: {delete_path}")

    try:
        # The partition key is 'file_name'
        partition_key = delete_path.split('-')[0]  # Extracting partition key from ID if needed
        delete_path = delete_path
        delete_result = delete_document_from_cosmos(delete_path, partition_key)

        if delete_result["status"]:
            if LOG_FLAG:
                logger.info(f"Successfully deleted document with ID: {delete_path}")
            return {"status": True}
        else:
            raise HTTPException(status_code=404, detail=f"Document not found: {delete_path}")
    
    except Exception as e:
        if LOG_FLAG:
            logger.error(f"[dataprep - del] failed to delete document {delete_path}: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
from azure.cosmos import exceptions

def delete_document_from_cosmos(file_name: str, partition_key: str):
    """Delete a document from Cosmos DB by its file name."""
    try:
        # Query the container to find the document based on the file name
        query = f"SELECT * FROM c WHERE c.file_name = @file_name"
        parameters = [{"name": "@file_name", "value": file_name}]
        #print(query)
        items = list(container.query_items(query=query, parameters=parameters, enable_cross_partition_query=True))
        print(items)
        # If the document is found, delete it
        if items:
            for item_to_delete in items:
                file_id = item_to_delete['id']
                # Deleting each document using the id and partition key
                container.delete_item(item=file_id, partition_key=partition_key)
            return {"status": True}
        else:
            return {"status": False, "message": "Document with the given file name not found."}
    
    except exceptions.CosmosResourceNotFoundError:
        return {"status": False, "message": "Document not found."}
    except exceptions.CosmosHttpResponseError as e:
        return {"status": False, "message": f"Cosmos DB error: {str(e)}"}
    except Exception as e:
        return {"status": False, "message": f"An error occurred: {str(e)}"}

# Helper function to encode the filename (you can modify this)
def encode_filename(filename: str) -> str:
    """Encode the filename by replacing spaces and converting it to lowercase."""
    return filename.replace(" ", "_").lower()

if __name__ == "__main__":
    opea_microservices["opea_service@prepare_doc_elastic"].start()

