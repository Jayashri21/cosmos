
import json
import os
from pathlib import Path
from typing import List, Optional, Union

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBED_MODEL,
    ES_CONNECTION_STRING,
    INDEX_NAME,
    LOG_FLAG,
    TEI_ENDPOINT,
    UPLOADED_FILES_PATH,
)
from elasticsearch import Elasticsearch
from fastapi import Body, File, Form, HTTPException, UploadFile
from langchain.text_splitter import HTMLHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.documents import Document
from langchain_elasticsearch import ElasticsearchStore
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings

from comps import CustomLogger, DocPath, opea_microservices, register_microservice
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

logger = CustomLogger(__name__)

cosmos_client = CosmosClient(COSMOS_CONNECTION_STRING)
database = cosmos_client.get_database_client(COSMOS_DB_NAME)
container = database.get_container_client(COSMOS_DB_CONTAINER)


def get_embedder() -> Union[HuggingFaceEndpointEmbeddings, HuggingFaceBgeEmbeddings]:
    """Obtain required Embedder."""

    if TEI_ENDPOINT:
        return HuggingFaceEndpointEmbeddings(model=TEI_ENDPOINT)
    else:
        return HuggingFaceBgeEmbeddings(model_name=EMBED_MODEL)


def get_elastic_store(embedder: Union[HuggingFaceEndpointEmbeddings, HuggingFaceBgeEmbeddings]) -> ElasticsearchStore:
    """Get Elasticsearch vector store."""

    return ElasticsearchStore(index_name=INDEX_NAME, embedding=embedder, es_connection=es_client)


# def search_by_filename(file_name: str) -> bool:
#     """Search Elasticsearch by file name."""

#     query = {"query": {"match": {"metadata.doc_name": {"query": file_name, "operator": "AND"}}}}
#     results = es_client.search(index=INDEX_NAME, body=query)

#     if LOG_FLAG:
#         logger.info(f"[ search by file ] searched by {file_name}")
#         logger.info(f"[ search by file ] {len(results['hits'])} results: {results}")

#     return results["hits"]["total"]["value"] > 0


def search_by_filename(file_name: str) -> bool:
    """Search Cosmos DB by file name."""
    
    query = f"SELECT * FROM c WHERE c.metadata.doc_name = '{file_name}'"
    results = list(container.query_items(query=query, enable_cross_partition_query=True))

    if LOG_FLAG:
        logger.info(f"[ search by file ] searched by {file_name}")
        logger.info(f"[ search by file ] {len(results)} results: {results}")

    return len(results) > 0




def ingest_doc_to_elastic(doc_path: DocPath) -> None:
    """Ingest documents to Elasticsearch."""

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

    batch_size = 32
    num_chunks = len(chunks)

    metadata = dict({"doc_name": str(file_name)})

    for i in range(0, num_chunks, batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_texts = batch_chunks

        documents = [Document(page_content=text, metadata=metadata) for text in batch_texts]
        _ = es_store.add_documents(documents=documents)
        if LOG_FLAG:
            logger.info(f"Processed batch {i // batch_size + 1}/{(num_chunks - 1) // batch_size + 1}")


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

            try:
                exists = search_by_filename(filename)
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed when searching in Elasticsearch for file {file.filename}.",
                )

            if exists:
                if LOG_FLAG:
                    logger.info(f"[ upload ] File {file.filename} already exists.")

                raise HTTPException(
                    status_code=400,
                    detail=f"Uploaded file {file.filename} already exists. Please change file name.",
                )

            await save_content_to_local_disk(save_path, file)

            ingest_doc_to_elastic(
                DocPath(
                    path=save_path,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    process_table=process_table,
                    table_strategy=table_strategy,
                )
            )
            if LOG_FLAG:
                logger.info(f"Successfully saved file {save_path}")

        result = {"status": 200, "message": "Data preparation succeeded"}

        if LOG_FLAG:
            logger.info(result)
        return result

    # if link_list:
    #     try:
    #         link_list = json.loads(link_list)  # Parse JSON string to list
    #         if not isinstance(link_list, list):
    #             raise HTTPException(status_code=400, detail="link_list should be a list.")

    #         await ingest_link_to_elastic(link_list)

    #         if LOG_FLAG:
    #             logger.info(f"Successfully saved link list {link_list}")

    #         result = {"status": 200, "message": "Data preparation succeeded"}

    #         if LOG_FLAG:
    #             logger.info(result)
    #         return result

    #     except json.JSONDecodeError:
    #         raise HTTPException(status_code=400, detail="Invalid JSON format for link_list.")

    raise HTTPException(status_code=400, detail="Must provide either a file or a string list.")



if __name__ == "__main__":
    es_client = Elasticsearch(hosts=ES_CONNECTION_STRING)
    es_store = get_elastic_store(get_embedder())
    create_upload_folder(UPLOADED_FILES_PATH)
    create_index()