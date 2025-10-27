from fastapi import APIRouter,Form
from fastapi.responses import JSONResponse
from modules.llm import get_llm_chain
from modules.query_handlers import query_chain
from langchain_core.documents import Document
# from langchain.schema import BaseRetriever
from langchain_core.retrievers import BaseRetriever

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone
from typing import List,Optional
from pydantic import Field
from logger import logger
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

router=APIRouter()
@router.post("/ask/")
async def ask_question(question:str=Form(...)):
    try:
        logger.info(f"User query:{question}")

        #embed model + pinecone setup
        pc=Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        index=pc.Index(os.environ["PINECONE_INDEX_NAME"])
        # embed_model=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        embed_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
        embedded_query=embed_model.embed_query(question)
        res=index.query(vector=embedded_query,top_k=5,include_metadata=True)

        docs=[
            Document(
                page_content=match["metadata"].get("text",""),
                metadata=match['metadata']


            ) for match in res["matches"]
        ]

        class SimpleRetriever(BaseRetriever):                          
            tags: Optional[List[str]] = Field(default_factory=list)
            metadata: Optional[dict] = Field(default_factory=dict)

            def __init__(self,documents: List[Document]):
                super().__init__()
                self._docs= documents

            def _get_relevant_documents(self,query: str)->List[Document]:
                return self._docs
            
        retriever=SimpleRetriever(docs)
        chain=get_llm_chain(retriever)
        result=query_chain(chain,question)
        
        logger.info("Query is successful")
        return result
    except Exception as e:
        logger.exception("Error processing in question")
        return JSONResponse(status_code=500,content={"error":str(e)})
    
   
   