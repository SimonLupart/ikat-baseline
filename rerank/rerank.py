from time import perf_counter
import json
from tqdm import tqdm
import torch
import argparse
from pyserini.search.lucene import LuceneSearcher
import os

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import CrossEncoder


def rerank_top_k(query_dict, top_k_results, tokenizer, cross_encoder, searcher, use_st):
    """
    Rerank the top 100 passages using the cross-encoder model.
    """
    reranked_results = {}

    for query_id, passage_scores in tqdm(top_k_results.items()):
        top_100_passages = sorted(passage_scores.items(), key=lambda x: x[1], reverse=True)[:100]
        passage_ids = [pid for pid, _ in top_100_passages]

        passage_text_mapping = searcher.batch_doc(passage_ids, threads=128)

        if use_st:
            query_passage_pairs = [
                [query_dict[query_id], json.loads(passage_text_mapping[pid].raw())['contents']]
                for pid in passage_ids if pid in passage_text_mapping
            ]
            scores = cross_encoder.predict(query_passage_pairs, batch_size=128, show_progress_bar=False)
        else:
            query_passage_pairs = []
            for pid in passage_ids:
                raw_passage = passage_text_mapping.get(pid)
                if raw_passage:
                    passage_text = json.loads(raw_passage.raw())['contents']
                    query_passage_pairs.append((query_dict[query_id], passage_text))

            scores = []
            for query_text, passage_text in tqdm(query_passage_pairs, desc="Reranking passages"):
                inputs = tokenizer(query_text, passage_text, return_tensors="pt", truncation=True, padding=True).to(cross_encoder.device)
                with torch.no_grad():
                    outputs = cross_encoder(**inputs)
                    scores.append(outputs.logits[0].item())

        passage_ids, scores = zip(*sorted(zip(passage_ids, scores), key=lambda x: x[1], reverse=True))
        reranked_results[query_id] = {pid: float(score) for pid, score in zip(passage_ids, scores)}


    return reranked_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rerank top-k passages using a cross-encoder model.")
    parser.add_argument("--index_path", type=str, required=True, help="Path to the Lucene index.")
    parser.add_argument("--model", type=str, default="naver/trecdl22-crossencoder-debertav3", help="Cross-encoder model name.")
    parser.add_argument("--run", type=str, required=True, help="Path to the JSON file containing top-k results.")
    parser.add_argument("--query_file", type=str, required=True, help="Path to the file containing queries.")
    parser.add_argument("--output", type=str, required=True, help="Path to save reranked results.")

    args = parser.parse_args()

    # Load top-k results
    with open(args.run, 'r') as f:
        top_k_results = json.load(f)

    # Load Lucene index
    print("Loading Lucene index...")
    t0 = perf_counter()
    searcher = LuceneSearcher(args.index_path)
    print("Index loaded in {:.2f}s".format(perf_counter() - t0))

    # Determine model type
    if "ms-marco" in args.model or "cross-encoder" in args.model:
        cross_encoder = CrossEncoder(args.model, device="cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = None
        use_st = True
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        cross_encoder = AutoModelForSequenceClassification.from_pretrained(args.model)
        cross_encoder.eval()
        cross_encoder.to("cuda" if torch.cuda.is_available() else "cpu")
        use_st = False

    # Load queries
    queries = {}
    with open(args.query_file, 'r') as f:
        for line in f:
            query_id, query_text = line.strip().split("\t")
            queries[query_id] = query_text
    print(f"Loaded {len(queries)} queries from {args.query_file}")

    # Rerank
    reranked_results = rerank_top_k(queries, top_k_results, tokenizer, cross_encoder, searcher, use_st)

    # Save output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(reranked_results, f)
    print(f"Saved reranked results to {args.output}")
