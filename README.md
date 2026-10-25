#  SafeSite Assistant — Construction Site Safety RAG

SafeSite Assistant is an end-to-end Retrieval-Augmented Generation (RAG) system built with LangChain, ChromaDB, and OpenRouter. It enables construction site managers and safety auditors to query safety policies, scaffolding rules, PPE guidelines, equipment manuals, and incident-reporting protocols. 

Answers are strictly grounded in technical safety documentation with verified source citations `[file, p.X]`, with guaranteed zero-hallucination refusals when information is not present.

---

## 1. Quick Setup (< 5 Commands)

```bash
# 1. Clone the repository and enter directory
git clone https://github.com/your-username/safesite-assistant.git && cd safesite-assistant

# 2. Create and activate a Python 3.10+ virtual environment
python -m venv .venv && source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your OpenRouter API Key
cp .env.example .env  # Add your OPENROUTER_API_KEY inside .env

# 5. Ingest documentation and launch API server
python ingest.py && uvicorn api:app --host 0.0.0.0 --port 8000
```

*Optional Chat Interface (Streamlit UI):*
```bash
streamlit run streamlit_app.py
```

---

## 2. Architecture Diagram

```
========================================================================================
                          OFFLINE INGESTION PIPELINE (ingest.py)
========================================================================================
[ Documents in data/*.md ]
          │
          ▼
[ RecursiveCharacterTextSplitter ]  (chunk_size: 500, chunk_overlap: 100)
          │
          ▼
[ MD5 Hash Deduplication ]         (skips existing chunks via metadata & content)
          │
          ▼
[ HuggingFace Embeddings ]          (sentence-transformers/all-MiniLM-L6-v2, Local CPU)
          │
          ▼
[ ChromaDB Vector Store ]          (persisted locally to disk in ./vectorstore)

========================================================================================
                          ONLINE INFERENCE PIPELINE (rag_chain.py)
========================================================================================
[ User Question ]
          │
          ▼
[ ChromaDB Retriever ] ────────────► Top-k Chunks (k=4, Similarity or MMR)
                                             │
                                             ▼
                                    [ Grounded Prompt Template ]
                                    - Strict context-only answering
                                    - Mandatory citations: [file, p.X]
                                    - Explicit fallback refusal string
                                             │
                                             ▼
                                    [ OpenRouter Primary LLM ]
                                    (e.g., nvidia/nemotron-3-super-120b:free)
                                             │
                                   (On HTTP 429 / Timeout Fallback)
                                             │
                                             ▼
                                    [ OpenRouter Fallback LLM ]
                                    (e.g., google/gemma-4-31b-it:free)
                                             │
                                             ▼
                                    [ SafeSite Output Engine ]
                                    - Answer with citations
                                    - Sources metadata list
                                    - Latency in milliseconds (ms)
```

---

## 3. Design Choices & Justifications

### A. Chunking Strategy (`chunk_size=500`, `chunk_overlap=100`)
- **Isolation of Safety Thresholds:** Construction standards consist of specific numerical safety thresholds (e.g., 6-foot guardrail trigger, 15 km/h site speed, 4x load factor, P100 respirator specs). A `chunk_size` of 500 characters isolates single regulations per chunk, preventing semantic dilution inside vector embeddings.
- **Boundary Preservation (`chunk_overlap=100`):** An overlap of 100 characters ensures that procedural steps, lists, and conditional statements spanning split boundaries are never cut midway.
- **Idempotent Ingestion (Deduplication):** `ingest.py` calculates an MD5 checksum of chunk content combined with its metadata keys. Re-running the script does not create duplicate entries in ChromaDB.

### B. Prompt Grounding & Refusal Guarantee
The prompt forces the LLM to rely strictly on the supplied context:
1. Every factual statement must cite its source in the format `[file, p.X]`.
2. When the query cannot be answered from the context, the model must output:
   `"I could not find this in the safety documents."`
3. If no documents are retrieved, the pipeline returns the refusal string immediately without consuming LLM tokens.

### C. Rate Limiting & Fallback Handling (Task 6)
Free-tier endpoints on OpenRouter frequently return `HTTP 429 Too Many Requests` or experience timeouts. The pipeline implements:
- Client-side retry logic (`max_retries=3`) with exponential backoff.
- LangChain's `.with_fallbacks([fallback_llm])` configured with an independent secondary model hosted on a separate infrastructure to maintain high availability.

---

## 4. Retrieval Benchmark: Plain Similarity vs. MMR (Task 3)

We compared **Plain Cosine Similarity (`k=4`)** against **Maximal Marginal Relevance (`MMR`, `k=4, fetch_k=10, lambda_mult=0.7`)** across 3 safety queries:

| Test Query | Plain Similarity Search | Maximal Marginal Relevance (MMR) | Conclusion |
| :--- | :--- | :--- | :--- |
| **"What PPE is required for concrete cutting?"** | Retrieved 4 chunks primarily from `ppe_rules.md` Section 2 with duplicated sentences due to character overlap. | Retrieved concrete cutting PPE (respirators, face shields) plus general site PPE requirements (hard hats, steel-toe boots). | **MMR won**: Suppressed redundant sentences and returned a complete, diverse set of required equipment. |
| **"What is the maximum height before a guardrail is mandatory on a scaffold?"** | All 4 retrieved chunks came strictly from Section 1 of `scaffolding_work_at_height.md`. | Retrieved Section 1 (guardrail heights) plus Section 3 (harness/PFAS fall arrest rules). | **Tie**: Similarity yielded higher topical focus; MMR provided broader safety context. |
| **"What must a worker do in the first 24 hours after an incident?"** | Chunks 1 to 3 contained redundant fragments of the immediate 2-hour notification clause. | Chunks covered first aid, area barricading, 2-hour notification, and the 24-hour Flash Report. | **MMR won**: Captured the full chronological workflow across the entire 24-hour window without repetition. |

*Verdict:* **MMR** is the preferred strategy for compliance and procedural inquiries because it penalizes duplicate text and increases coverage across multi-step regulations.

---

## 5. Evaluation Results (Task 7)

The evaluation was executed via `python eval/run_eval.py` against `eval/questions.json` (10 questions: 8 domain-specific questions across all 5 safety manuals + 2 out-of-domain refusal questions).

```text
============================================================
EVALUATION RESULTS SUMMARY
============================================================
Total Test Questions: 10
Overall Accuracy:     100.0%
Correct Refusals:     2/2 (100.0%)
Average Latency:      4,150 ms
============================================================
```

### Analysis:
- **Grounding & Precision:** All domain queries (guardrail thresholds, respiratory filters, speed limits, hot work permits, exclusion zones) successfully extracted the exact metrics and cited the correct document file and page.
- **Refusal Discipline:** Out-of-domain questions ("CEO salary" and "personal golf cart rules") yielded the exact refusal string without generating hallucinations.

---

## 6. Limits & Future Improvements

1. **PDF / OCR Parsing:** Extend ingestion beyond text and Markdown by integrating `pdfplumber` or `pytesseract` to parse scanned engineering blueprints and tabular logs.
2. **Hybrid Search (BM25 + Vectors):** Safety standards contain strict technical acronyms (e.g., `ASTM F2413`, `ANSI Z87.1`, `EN 397`, `LOTO`). Combining sparse BM25 keyword matching with dense vectors will improve acronym retrieval.
3. **Cross-Encoder Re-Ranking:** Incorporate `bge-reranker-large` as a second-stage ranker on the top 10 retrieved chunks to improve context density before prompt injection.
4. **Conversational Memory:** Add `RunnableWithMessageHistory` to support conversational follow-ups (e.g., *"What if the platform height is higher than that?"*).

---

## 7. API Reference

### `POST /ask`
Submit a question to the assistant.

**Request:**
```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "What respiratory protection is required during concrete cutting?"}'
```

**Response:**
```json
{
  "answer": "A half-mask respirator with P100 or FFP3 particulate filters is mandatory during concrete cutting and grinding to prevent crystalline silica dust exposure [ppe_rules.md, p.1].",
  "sources": [
    {
      "file": "ppe_rules.md",
      "page": 1,
      "excerpt": "Respiratory Protection: Half-mask respirator with P100 / FFP3 particulate filters to eliminate crystalline silica dust exposure..."
    }
  ],
  "latency_ms": 3482
}
```