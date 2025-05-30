import argparse
import json
from tqdm import tqdm
import os
from pyserini.search.lucene import LuceneSearcher


def run_bm25(searcher, query_text, k):
    hits = searcher.search(query_text, k=k)
    return {hit.docid: float(hit.score) for hit in hits}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", required=True, help="Path to Lucene index")
    parser.add_argument("--topics", required=True, help="Path to topic TSV file")
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    parser.add_argument("--k", type=int, default=1000, help="Number of passages to return")

    args = parser.parse_args()

    searcher = LuceneSearcher(args.index_path)

    ranking = {}
    with open(args.topics) as f:
        for line in tqdm(f):
            splitted = line.strip().split("\t")
            if len(splitted) < 2:
                qid = splitted[0]
                query_text = " "
            else:
                qid, query_text = splitted
            ranking[qid] = run_bm25(searcher, query_text, args.k)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(ranking, f)


if __name__ == "__main__":
    main()
