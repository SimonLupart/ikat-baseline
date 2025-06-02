import hydra
from omegaconf import DictConfig
import os
import gc
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pyserini.search import LuceneSearcher
from time import perf_counter
from tqdm import tqdm

from splade.conf.CONFIG_CHOICE import CONFIG_NAME, CONFIG_PATH
from splade.evaluation.datasets import CollectionDataLoader, CollectionDatasetPreLoad
from splade.evaluation.models import Splade
from splade.evaluation.transformer_evaluator import SparseRetrieval
from splade.evaluation.utils.utils import get_initialize_config


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base="1.2")
def retrieve_evaluate(exp_dict: DictConfig):
    exp_dict, config, init_dict, model_training_config = get_initialize_config(exp_dict)

    # If HF: need to update config.
    if "hf_training" in config and config["hf_training"]:
        init_dict.model_type_or_dir = os.path.join(config.checkpoint_dir, "model")
        init_dict.model_type_or_dir_q = os.path.join(config.checkpoint_dir, "model/query") if init_dict.model_type_or_dir_q else None

    model = Splade(**init_dict)

    # Load Lucene index using Pyserini
    lucene_index_path = config["lucene_index"]  # Update with your index path
    print("Loading Lucene index...")
    t0 = perf_counter()
    searcher = LuceneSearcher(lucene_index_path)
    print("Loading index took {:.3f} sec".format(perf_counter() - t0))

    # Initialize SparseRetrieval once
    evaluator = SparseRetrieval(config=config, model=model, compute_stats=True, dim_voc=model.output_dim)

    print("Index loaded. Ready for interactive retrieval.")

    # Load the cross-encoder model for reranking
    cross_encoder_name = "naver/trecdl22-crossencoder-debertav3"
    tokenizer = AutoTokenizer.from_pretrained(cross_encoder_name)
    cross_encoder = AutoModelForSequenceClassification.from_pretrained(cross_encoder_name)
    cross_encoder.eval()
    cross_encoder.to("cuda" if torch.cuda.is_available() else "cpu")

    try:
        current_query = None
        reranked_passages = []
        passage_index = 0

        while True:
            user_query = input("\n\nEnter your query (or type 'exit' to quit, 'n' to show more results):\n").strip()

            if user_query.lower() == "exit":
                print("Exiting interactive retrieval.")
                break

            if user_query.lower() == "n":
                if not reranked_passages:
                    print("No results to show. Please enter a query first.")
                    continue

                # Show next 10 results
                next_passages = reranked_passages[passage_index:passage_index+10]
                if not next_passages:
                    print("No more results.")
                    continue

                for i, passage in enumerate(next_passages, start=passage_index + 1):
                    print("--------------------")
                    print(f"Rank {i}: (Score: {passage['rerank_score']})")
                    print(f"Passage ID: {passage['id']}")
                    print(f"{passage['text']}\n")
                passage_index += 10
                continue

            # New query
            current_query = user_query
            reranked_passages = []
            passage_index = 0

            # Create a temporary query collection
            temp_query_path = config["temp_path_q"]
            with open(temp_query_path, "w") as f:
                f.write(f"0\t{user_query}\n")

            q_collection = CollectionDatasetPreLoad(data_dir=temp_query_path, id_style="row_id")
            q_loader = CollectionDataLoader(dataset=q_collection, tokenizer_type=model_training_config["tokenizer_type"],
                                            max_length=model_training_config["max_length"], batch_size=1,
                                            shuffle=False, num_workers=1)

            # Perform retrieval
            top_k_results = evaluator.retrieve(q_loader, top_k=exp_dict["config"]["top_k"],
                                               dataset_name="ikat2025", return_d=True,
                                               threshold=exp_dict["config"]["threshold"])

            reranked_passages = rerank_top_k(user_query, top_k_results["retrieval"], tokenizer, cross_encoder, searcher)

            # Show first 10
            for i, passage in enumerate(reranked_passages[:10], start=1):
                print("--------------------")
                print(f"Rank {i}: (Score: {passage['rerank_score']})")
                print(f"Passage ID: {passage['id']}")
                print(f"{passage['text']}\n")
            passage_index = 10

    except KeyboardInterrupt:
        print("\nInteractive session terminated by user.")

    # Cleanup
    evaluator = None
    gc.collect()
    torch.cuda.empty_cache()


def rerank_top_k(query, top_k_results, tokenizer, cross_encoder, searcher):
    """
    Rerank the top 100 passages using the cross-encoder model and return the sorted list.
    """
    all_reranked = []

    for query_id, passage_scores in top_k_results.items():
        top_100_passages = sorted(passage_scores.items(), key=lambda x: x[1], reverse=True)[:100]
        passage_ids = [pid for pid, _ in top_100_passages]

        t0 = perf_counter()
        passage_text_mapping = searcher.batch_doc(passage_ids, threads=128)
        print("Retrieving passage text took {:.3f} sec".format(perf_counter() - t0))

        query_passage_pairs = []
        for pid in passage_ids:
            raw_passage = passage_text_mapping.get(pid)
            if raw_passage:
                passage_text = json.loads(raw_passage.raw())['contents']
                query_passage_pairs.append((query, passage_text))

        scores = []
        for query_text, passage_text in tqdm(query_passage_pairs, desc="Reranking passages"):
            inputs = tokenizer(query_text, passage_text, return_tensors="pt", truncation=True, padding=True).to(cross_encoder.device)
            with torch.no_grad():
                outputs = cross_encoder(**inputs)
                scores.append(outputs.logits[0].item())

        passages = [
            {"id": pid, "text": json.loads(passage_text_mapping[pid].raw())['contents'], "rerank_score": scores[i]}
            for i, pid in enumerate(passage_ids)
        ]
        passages.sort(key=lambda x: x["rerank_score"], reverse=True)
        all_reranked.extend(passages)

    return all_reranked

    # Cleanup
    evaluator = None
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    retrieve_evaluate()
