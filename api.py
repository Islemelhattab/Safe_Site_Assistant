import os
import time
from typing import Any, Dict, List
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from rag_chain import SafeSiteRAG

app = FastAPI(
    title="SafeSite Assistant API",
    description="Assistant de sécurité sur chantier de construction.",
    version="1.0.0"
)

# Initialisation du moteur RAG
rag_engine = SafeSiteRAG(search_type="similarity")


class QueryRequest(BaseModel):
    # Remplacement de 'example' par 'json_schema_extra' pour Pydantic v2
    question: str = Field(
        ..., 
        json_schema_extra={"example": "What is the maximum height before a guardrail is mandatory on a scaffold?"}
    )


class SourceItem(BaseModel):
    file: str
    page: int
    excerpt: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    latency_ms: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=QueryResponse, status_code=status.HTTP_200_OK)
def ask_question(payload: QueryRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        result = rag_engine.query(payload.question)
        return result
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while generating response: {str(exc)}"
        )


if __name__ == "__main__":
    import uvicorn
    # IMPORTANT: "api:app" SANS le ".py"
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)