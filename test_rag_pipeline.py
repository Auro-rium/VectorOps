# test_rag_final.py
import sys
from rag_pipeline import build_index_for_folder, query_and_print
import os

def main(folder="./data"):
    idx, emb, chunks = build_index_for_folder(folder)
    metadata = [{"id":c.id, "text":c.text, "metadata": c.metadata} for c in chunks]

    sample_order=None
    for m in metadata:
        if "Order_ID" in m["metadata"]:
            sample_order=str(m["metadata"]["Order_ID"])
            break

    queries=[]
    if sample_order:
        queries.append(f"Order_ID: {sample_order}")
    queries += ["delay OR late OR delayed", "freight cost rate carrier"]

    for q in queries:
        query_and_print(idx, emb, metadata, q, k=5)

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv)>1 else "./data"
    if not os.path.isdir(folder):
        print("Folder not found:", folder); raise SystemExit(1)
    main(folder)
