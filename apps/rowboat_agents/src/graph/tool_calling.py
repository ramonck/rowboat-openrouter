from bson.objectid import ObjectId
from openai import OpenAI
import os
from motor.motor_asyncio import AsyncIOMotorClient
import asyncio
from dataclasses import dataclass
from typing import Dict, List, Any
from qdrant_client import QdrantClient
import json
# Initialize MongoDB client
mongo_uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
mongo_client = AsyncIOMotorClient(mongo_uri)
db = mongo_client.rowboat
data_sources_collection = db['sources']
data_source_docs_collection = db['source_docs']


qdrant_client = QdrantClient(
    url=os.environ.get("QDRANT_URL"),
    api_key=os.environ.get("QDRANT_API_KEY") or None
)
# Initialize OpenAI client
client = OpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# Define embedding model - ensure this is a model supported by a provider on OpenRouter
embedding_model = "openai/text-embedding-3-small"  # Explicitly prefix with "openai/"

async def embed(model: str, value: str) -> dict:
    """
    Generate embeddings using OpenAI's embedding models.

    Args:
        model (str): The embedding model to use (e.g., "text-embedding-3-small").
        value (str): The text to embed.

    Returns:
        dict: A dictionary containing the embedding.
    """
    response = client.embeddings.create(
        model=model,
        input=value
    )
    return {"embedding": response.data[0].embedding}

async def call_rag_tool(
    project_id: str,
    query: str,
    source_ids: list[str],
    return_type: str,
    k: int,
) -> dict:
    """
    Runs the RAG tool call to retrieve information based on the query and source IDs.

    Args:
        project_id (str): The ID of the project.
        query (str): The query string to search for.
        source_ids (list[str]): List of source IDs to filter the search.
        return_type (str): The type of return, e.g., 'chunks' or other.
        k (int): The number of results to return.

    Returns:
        dict: A dictionary containing the results of the search.
    """

    print("\n\n calling rag tool \n\n")
    print(query)
    # Create embedding for the query
    embed_result = await embed(model=embedding_model, value=query)

    # print(embed_result)
    # Fetch all active data sources for this project
    sources = await data_sources_collection.find({
        "projectId": project_id,
        "active": True
    }).to_list(length=None)

    print(sources)
    # Filter sources to those in source_ids
    valid_source_ids = [
        str(s["_id"]) for s in sources if str(s["_id"]) in source_ids
    ]

    print(valid_source_ids)
    # If no valid sources are found, return empty results
    if not valid_source_ids:
        return ''

    # Perform Qdrant vector search
    qdrant_results = qdrant_client.search(
        collection_name="embeddings",
        query_vector=embed_result["embedding"],
        query_filter={
            "must": [
                {"key": "projectId", "match": {"value": project_id}},
                {"key": "sourceId", "match": {"any": valid_source_ids}},
            ]
        },
        limit=k,
        with_payload=True
    )

    # Map the Qdrant results to the desired format
    results = [
        {
            "title": point.payload["title"],
            "name": point.payload["name"],
            "content": point.payload["content"],
            "docId": point.payload["docId"],
            "sourceId": point.payload["sourceId"],
        }
        for point in qdrant_results
    ]

    print(return_type)
    print(results)
    # If return_type is 'chunks', return the results directly
    if return_type == "chunks":
        return json.dumps({"Information": results}, indent=2)

    # Otherwise, fetch the full document contents from MongoDB
    doc_ids = [ObjectId(r["docId"]) for r in results]
    docs = await data_source_docs_collection.find({"_id": {"$in": doc_ids}}).to_list(length=None)

    # Create a dictionary for quick lookup of documents by their string ID
    doc_dict = {str(doc["_id"]): doc for doc in docs}

    # Update the results with the full document content
    results = [
        {**r, "content": doc_dict.get(r["docId"], {}).get("content", "")}
        for r in results
    ]

    # Convert results to a JSON string
    formatted_string = json.dumps({"Information": results}, indent=2)
    print(formatted_string)
    return formatted_string


if __name__ == "__main__":
    asyncio.run(call_rag_tool(
        project_id="faf2bfb3-41d4-4299-b0d2-048581ea9bd8",
        query="What is the range on your scooter",
        source_ids=["67e102c9fab4514d7aaeb5a4"],
        return_type="docs",
        k=3))