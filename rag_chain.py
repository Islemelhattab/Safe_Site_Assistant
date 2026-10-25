import os
import time
from typing import Any, Dict, List, Tuple
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

PERSIST_DIR = os.getenv("VECTORSTORE_DIR", "vectorstore")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "poolside/laguna-s-2.1:free")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")


def get_vectorstore() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)


def format_context_and_extract_sources(docs: List[Document]) -> Tuple[str, List[Dict[str, Any]]]:
    """Extrait le texte et les métadonnées de source pour le prompt et la réponse."""
    context_blocks = []
    sources = []
    for doc in docs:
        file_name = doc.metadata.get("file", "unknown")
        page = doc.metadata.get("page", 1)
        tag = f"[{file_name}, p.{page}]"
        context_blocks.append(f"{tag}\n{doc.page_content.strip()}")
        sources.append({
            "file": file_name,
            "page": page,
            "excerpt": doc.page_content.strip().replace("\n", " ")[:150] + "..."
        })
    return "\n\n".join(context_blocks), sources


SYSTEM_PROMPT = (
    "You are SafeSite Assistant, a strict construction safety compliance AI.\n"
    "Answer the user's question using ONLY the context provided below.\n"
    "Rules:\n"
    "1. Cite sources immediately after factual statements using the exact format: [file, p.X].\n"
    "2. If the answer cannot be deduced strictly from the context, respond EXACTLY with:\n"
    "   \"I could not find this in the safety documents.\"\n"
    "3. Never make assumptions, use general knowledge, or speculate.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)

prompt_template = ChatPromptTemplate.from_template(SYSTEM_PROMPT)

# Modèle principal avec retry et timeout
primary_llm = ChatOpenAI(
    model=PRIMARY_MODEL,
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    temperature=0.0,
    max_retries=3,
    request_timeout=30.0,
    default_headers={"HTTP-Referer": "https://safesite.local", "X-Title": "SafeSite Assistant"},
)

fallback_llm = ChatOpenAI(
    model=FALLBACK_MODEL,
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    temperature=0.0,
    max_retries=2,
    request_timeout=20.0,
    default_headers={"HTTP-Referer": "https://safesite.local", "X-Title": "SafeSite Assistant"},
)

robust_llm = primary_llm.with_fallbacks([fallback_llm])


class SafeSiteRAG:
    def __init__(self, search_type: str = "similarity"):
        db = get_vectorstore()
        if search_type == "mmr":
            self.retriever = db.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.7}
            )
        else:
            self.retriever = db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )

        self.generation_chain = (
            prompt_template
            | robust_llm
            | StrOutputParser()
        )

    def query(self, question: str) -> Dict[str, Any]:
        start_time = time.perf_counter()
        retrieved_docs = self.retriever.invoke(question)
        context_str, sources = format_context_and_extract_sources(retrieved_docs)

        answer = self.generation_chain.invoke({
            "context": context_str,
            "question": question
        })
        latency_ms = int((time.perf_counter() - start_time) * 1000)

        # Si aucun document trouvé ou refus explicite du modèle
        if not retrieved_docs or "I could not find this in the safety documents." in answer:
            return {
                "answer": "I could not find this in the safety documents.",
                "sources": [],
                "latency_ms": latency_ms,
            }

        return {
            "answer": answer.strip(),
            "sources": sources,
            "latency_ms": latency_ms,
        }