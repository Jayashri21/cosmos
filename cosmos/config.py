# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

COSMOS_CONNECTION_STRING = os.getenv("COSMOS_CONNECTION_STRING", "")
UPLOADED_FILES_PATH = os.getenv("UPLOADED_FILES_PATH", "./uploaded_files/")


endpoint =  os.getenv("endpoint" , "")

key =  os.getenv("key" , "")

# Embedding model
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN" , "" )

# TEI Embedding endpoints
TEI_ENDPOINT = os.getenv("TEI_ENDPOINT", "")

# Vector Index Configuration
INDEX_NAME = os.getenv("INDEX_NAME", "rag-elastic")

# chunk parameters
CHUNK_SIZE = os.getenv("CHUNK_SIZE", 1500)
CHUNK_OVERLAP = os.getenv("CHUNK_OVERLAP", 100)

# Logging enabled
LOG_FLAG = os.getenv("LOGFLAG", False)

COSMOS_DB_NAME = os.getenv("COSMOS_DB_NAME" , "DB_new")
COSMOS_DB_CONTAINER = os.getenv("COSMOS_DB_CONTAINER" , "container_new")

