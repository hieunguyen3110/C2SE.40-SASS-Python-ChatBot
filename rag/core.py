import pymongo
import google.generativeai as genai
from IPython.display import Markdown
import textwrap
from embeddings import SentenceTransformerEmbedding, EmbeddingConfig

class RAG:
    def __init__(self, 
            mongodbUri: str,
            dbName: str,
            dbCollection: str,
            llm,
            # embeddingName: str ='keepitreal/vietnamese-sbert',
            embeddingName: str = 'sentence-transformers/all-mpnet-base-v2',
        ):
        self.client = pymongo.MongoClient(mongodbUri)
        self.db = self.client[dbName] 
        self.collection = self.db[dbCollection]
        self.embedding_model = SentenceTransformerEmbedding(
            EmbeddingConfig(name=embeddingName)
        )
        self.llm = llm

    def get_embedding(self, text):
        if not text.strip():
            return []

        embedding = self.embedding_model.encode(text)
        return embedding.tolist()

    def vector_search(
            self, 
            user_query: str, 
            limit=4):
        """
        Perform a vector search in the MongoDB collection based on the user query.

        Args:
        user_query (str): The user's query string.

        Returns:
        list: A list of matching documents.
        """

        # Generate embedding for the user query
        query_embedding = self.get_embedding(user_query)

        if query_embedding is None:
            return "Invalid query or embedding generation failed."

        # Define the vector search pipeline
        vector_search_stage = {
            "$vectorSearch": {
                "index": "vector_index",
                "queryVector": query_embedding,
                "path": "embeddings",
                "numCandidates": 500,
                "limit": limit,
            }
        }

        unset_stage = {
            "$unset": "embeddings"
        }

        project_stage = {
            "$project": {
                "_id": 0,  
                "title": 1,
                # "product_specs": 1,
                "content": 1,
                "file_name":1,
                "score": {
                    "$meta": "vectorSearchScore"
                }
            }
        }

        pipeline = [vector_search_stage, unset_stage, project_stage]

        # Execute the search
        results = self.collection.aggregate(pipeline)
        return list(results)

    def enhance_prompt(self, query,file_source):
        get_knowledge = self.vector_search(query.text, 10)
        enhanced_prompt = ""
        i = 0
        for result in get_knowledge:
            if result.get('title'):
                i += 1
                enhanced_prompt += f"\n {i}) tiêu đề: {result.get('title')}"
                if result.get('file_name'):
                    file_source.append(result.get('file_name'))
                    print(f"Appended file_name: {result.get('file_name')}")  # Debug
                    enhanced_prompt += f", tham chiếu từ tài liệu : {result.get('file_name')}"

                if result.get('content'):
                    enhanced_prompt += f", nội dung : {result.get('content')}"
                else:
                    # Mock up data
                    # Retrieval model pricing from the internet.
                    enhanced_prompt += f", hiện tại, tôi chưa biết về câu hỏi"

        return enhanced_prompt

    def generate_content(self, prompt):
        return self.llm.generate_content(prompt)

    def _to_markdown(text):
        text = text.replace('•', '  *')
        return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))
