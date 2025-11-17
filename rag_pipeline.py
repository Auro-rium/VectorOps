# rag_pipeline_final.py
"""
Final RAG pipeline using Vertex embeddings (robust client-or-REST fallback).
Drop CSVs into a folder and run the test runner.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import os, json, hashlib, time
import pandas as pd
import numpy as np
import faiss
import requests
from dotenv import load_dotenv
from tqdm import tqdm

# auth helpers
import google.auth
from google.auth.transport.requests import Request as GoogleRequest

load_dotenv()

# --------- CONFIG ----------
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")        # optional but recommended
GCP_REGION = os.getenv("GCP_REGION", "us-central1")
VERTEX_API_KEY = os.getenv("VERTEX_API_KEY", None)  # optional
# REST URL uses explicit project for billing; if GCP_PROJECT_ID missing, we will use projects/- fallback
REST_MODEL_URL_TEMPLATE = "https://{region}-aiplatform.googleapis.com/v1/projects/{project_part}/locations/{region}/publishers/google/models/gemini-embedding-001:predict"

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "256"))
EMB_CACHE_DIR = os.getenv("EMB_CACHE_DIR", "emb_cache")
INDEX_DIR = os.getenv("INDEX_DIR", "faiss_index")
METADATA_FILE = os.getenv("METADATA_FILE", "index_metadata.json")
CHUNK_TOKEN_TARGET = int(os.getenv("CHUNK_TOKEN_TARGET", "200"))
TOP_K = int(os.getenv("TOP_K", "5"))

os.makedirs(EMB_CACHE_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# ---------- Data class ----------
@dataclass
class Chunk:
    id: str
    source: str
    text: str
    metadata: Dict[str, Any]
    embedding: List[float] = None

# ---------- Utilities ----------
def chunk_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def hash_id(text: str) -> str:
    return hashlib.sha1(text.encode('utf-8')).hexdigest()[:12]

def l2_normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1e-6
    return v / norms

# ---------- Ingest & chunk ----------
def ingest_csvs_from_folder(folder: str) -> List[Dict[str,Any]]:
    files = [f for f in sorted(os.listdir(folder)) if f.lower().endswith(".csv")]
    rows = []
    for fn in files:
        fp = os.path.join(folder, fn)
        try:
            df = pd.read_csv(fp)
        except Exception:
            df = pd.read_csv(fp, encoding="latin1")
        src = fn
        for _, r in df.iterrows():
            row = r.dropna().to_dict()
            row["_source_file"] = src
            rows.append(row)
    return rows

def row_to_text(row: Dict[str,Any], max_len=1000) -> str:
    parts = []
    priority_fields = ["Order_ID","Order_Date","Orig_Port","Dest_Port","Product_ID","Customer","Carrier","Cost","Rate","TPT_Day_Count","Ship_Late_Day_Count"]
    for f in priority_fields:
        if f in row:
            parts.append(f"{f}: {row[f]}")
    if len(parts) < 4:
        for k,v in row.items():
            if k.startswith("_") or k in priority_fields:
                continue
            parts.append(f"{k}: {v}")
            if len(" ".join(parts)) > max_len:
                break
    return " | ".join(parts)

def naive_chunk_text(text: str, target_words=CHUNK_TOKEN_TARGET) -> List[str]:
    sents = text.split(". ")
    chunks=[]
    cur=[]
    cur_words=0
    for s in sents:
        w = len(s.split())
        if cur_words + w > target_words and cur:
            chunks.append(". ".join(cur).strip())
            cur=[s]; cur_words=w
        else:
            cur.append(s); cur_words += w
    if cur:
        chunks.append(". ".join(cur).strip())
    return chunks

def build_chunks_from_rows(rows: List[Dict[str,Any]]) -> List[Chunk]:
    chunks=[]
    for r in rows:
        src = r.get("_source_file","unknown")
        text = row_to_text(r)
        parts = naive_chunk_text(text, target_words=CHUNK_TOKEN_TARGET)
        for i,p in enumerate(parts):
            cid = hash_id(f"{src}-{i}-{p[:80]}")
            metadata = {k:v for k,v in r.items() if not k.startswith("_")}
            metadata.update({"source_file": src, "row_preview": p[:200], "part_index": i})
            chunks.append(Chunk(id=cid, source=src, text=p, metadata=metadata))
    return chunks

# ---------- Embedder (tries client PublisherModel, then REST) ----------
class VertexEmbedder:
    def __init__(self, batch_size=BATCH_SIZE, region=GCP_REGION, project=GCP_PROJECT_ID, api_key=VERTEX_API_KEY):
        self.batch_size = batch_size
        self.region = region
        self.project = project
        self.api_key = api_key
        self.client_mode = False
        self.model_client = None
        self.dim = None

        # try client PublisherModel if available
        try:
            from google.cloud import aiplatform
            # if publisher API present, use it
            if hasattr(aiplatform, "PublisherModel"):
                self.model_client = aiplatform.PublisherModel.from_pretrained("google/gemini-embedding-001")
                self.client_mode = True
            else:
                # older versions may not expose PublisherModel; fall through to REST
                self.client_mode = False
        except Exception:
            self.client_mode = False

        # prepare REST URL if needed
        project_part = self.project if self.project else "-"
        self.rest_url = REST_MODEL_URL_TEMPLATE.format(region=self.region, project_part=project_part)

    # parse various shapes into list of floats
    def _flatten_pred(self, pred) -> List[float]:
        if isinstance(pred, list) and all(isinstance(x,(int,float)) for x in pred):
            return pred
        if isinstance(pred, dict):
            # common: {"embedding":{"values":[..]}} or {"values":[..]}
            if "embedding" in pred and isinstance(pred["embedding"], dict) and "values" in pred["embedding"]:
                return pred["embedding"]["values"]
            if "values" in pred and isinstance(pred["values"], list):
                return pred["values"]
        # recursive flatten
        out=[]
        def walk(x):
            if isinstance(x,(int,float)): out.append(x)
            elif isinstance(x,(list,tuple)): 
                for e in x: walk(e)
            elif isinstance(x,dict):
                for v in x.values(): walk(v)
        walk(pred)
        return out

    # client-mode embedding
    def embed_batch_client(self, texts: List[str]) -> np.ndarray:
        instances = [{"content": t} for t in texts]
        resp = self.model_client.predict(instances=instances)
        preds = list(resp.predictions)
        vecs = []
        for p in preds:
            v = self._flatten_pred(p)
            vecs.append(np.array(v, dtype=np.float32))
        out = np.vstack(vecs)
        if self.dim is None:
            self.dim = out.shape[1]
        return out

    # rest-mode embedding (uses API key if present, otherwise ADC token)
    def _get_adc_token(self) -> str:
        creds, _ = google.auth.default()
        creds.refresh(GoogleRequest())
        return creds.token

    def embed_batch_rest(self, texts: List[str]) -> np.ndarray:
        headers = {"Content-Type":"application/json"}
        # pick auth method
        if self.api_key:
            headers["x-goog-api-key"] = self.api_key
            # send user project if available
            if self.project:
                headers["x-goog-user-project"] = self.project
        else:
            token = self._get_adc_token()
            headers["Authorization"] = f"Bearer {token}"
            if self.project:
                headers["x-goog-user-project"] = self.project

        instances = [{"content": t} for t in texts]
        payload = {"instances": instances, "parameters": {}}
        r = requests.post(self.rest_url, headers=headers, json=payload, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"Vertex REST error {r.status_code}: {r.text}")
        resp = r.json()
        preds = resp.get("predictions", [])
        vecs = []
        for p in preds:
            v = self._flatten_pred(p)
            vecs.append(np.array(v, dtype=np.float32))
        out = np.vstack(vecs)
        if self.dim is None:
            self.dim = out.shape[1]
        return out

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        all = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            if self.client_mode:
                embs = self.embed_batch_client(batch)
            else:
                embs = self.embed_batch_rest(batch)
            all.append(embs)
            time.sleep(0.01)
        return np.vstack(all).astype(np.float32)

# ---------- caching wrapper ----------
def embed_with_cache(embedder: VertexEmbedder, texts: List[str]) -> np.ndarray:
    results = [None]*len(texts)
    to_call_idx=[]
    to_call_texts=[]
    for i,t in enumerate(texts):
        h = chunk_hash(t)
        cf = os.path.join(EMB_CACHE_DIR, f"{h}.npy")
        if os.path.exists(cf):
            results[i] = np.load(cf)
        else:
            to_call_idx.append(i)
            to_call_texts.append(t)
    if to_call_texts:
        new_embs = embedder.embed_texts(to_call_texts)
        for j, idx in enumerate(to_call_idx):
            vec = new_embs[j].astype(np.float32)
            results[idx] = vec
            np.save(os.path.join(EMB_CACHE_DIR, f"{chunk_hash(to_call_texts[j])}.npy"), vec)
    if len(results)==0:
        return np.zeros((0,1), dtype=np.float32)
    return np.vstack(results).astype(np.float32)

# ---------- FAISS + metadata ----------
def build_faiss_index(embeddings: np.ndarray, index_path: str=None) -> faiss.Index:
    if embeddings.shape[0] == 0:
        raise RuntimeError("No embeddings to index")
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    embeddings = l2_normalize(embeddings)
    index.add(embeddings)
    if index_path:
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(index, index_path)
    return index

def save_metadata(chunks: List[Chunk], path=METADATA_FILE):
    meta=[]
    for c in chunks:
        meta.append({"id":c.id, "source":c.source, "text":c.text, "metadata":c.metadata})
    with open(path,"w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def load_metadata(path=METADATA_FILE):
    with open(path,"r", encoding="utf-8") as f:
        return json.load(f)

# ---------- retrieve ----------
def retrieve(index: faiss.Index, embedder: VertexEmbedder, query: str, metadata: List[Dict[str,Any]], k=TOP_K):
    q_emb = embedder.embed_texts([query])
    q_emb = l2_normalize(q_emb)
    D, I = index.search(q_emb, k)
    results=[]
    for idx, score in zip(I[0].tolist(), D[0].tolist()):
        if idx < 0 or idx >= len(metadata):
            continue
        e = metadata[idx].copy()
        e["score"] = float(score)
        results.append(e)
    return results

# ---------- workflow ----------
def build_index_for_folder(folder: str, index_out=os.path.join(INDEX_DIR,"faiss.index"), meta_out=METADATA_FILE):
    rows = ingest_csvs_from_folder(folder)
    if len(rows)==0:
        raise RuntimeError(f"No CSVs in folder: {folder}")
    print("Ingested rows:", len(rows))
    chunks = build_chunks_from_rows(rows)
    print("Built chunks:", len(chunks))
    texts = [c.text for c in chunks]

    embedder = VertexEmbedder(batch_size=BATCH_SIZE, region=GCP_REGION, project=GCP_PROJECT_ID, api_key=VERTEX_API_KEY)
    print("Embedding chunks (with cache)...")
    embeddings = embed_with_cache(embedder, texts)
    print("Embeddings shape:", embeddings.shape)

    for c, e in zip(chunks, embeddings):
        c.embedding = e.tolist()

    index = build_faiss_index(embeddings, index_out)
    save_metadata(chunks, meta_out)
    print("Index saved:", index.ntotal)
    return index, embedder, chunks

# ---------- simple fact-check ----------
def simple_fact_check(answer_text: str, retrieved: List[Dict[str,Any]]):
    tokens = set([t for t in answer_text.split() if any(c.isdigit() for c in t)])
    passage_text = " ".join([r["text"] for r in retrieved])
    issues=[]
    for t in tokens:
        if t not in passage_text:
            issues.append({"token":t, "present":False})
    return {"ok": len(issues)==0, "issues": issues}

# ---------- helpers ----------
def query_and_print(index, embedder, metadata, query, k=TOP_K):
    res = retrieve(index, embedder, query, metadata, k=k)
    print("Query:", query)
    for r in res:
        print(f"- score={r['score']:.3f} source={r['source']} preview={r['text'][:120]}")
    return res

# End of rag_pipeline_final.py
