# import hydra
# from omegaconf import DictConfig
# import os
# import gc
# import json
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from pyserini.search import LuceneSearcher
# from time import perf_counter
# from tqdm import tqdm

# from splade.conf.CONFIG_CHOICE import CONFIG_NAME, CONFIG_PATH

# from splade.evaluation.datasets import CollectionDataLoader, CollectionDatasetPreLoad
# from splade.evaluation.models import Splade
# from splade.evaluation.transformer_evaluator import SparseRetrieval
# from splade.evaluation.utils.utils import get_initialize_config

# from tqdm import tqdm


# @hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base="1.2")
# def retrieve_evaluate(exp_dict: DictConfig):
#     exp_dict, config, init_dict, model_training_config = get_initialize_config(exp_dict)

#     # If HF: need to update config.
#     if "hf_training" in config and config["hf_training"]:
#         init_dict.model_type_or_dir = os.path.join(config.checkpoint_dir, "model")
#         init_dict.model_type_or_dir_q = os.path.join(config.checkpoint_dir, "model/query") if init_dict.model_type_or_dir_q else None

#     model = Splade(**init_dict)

#     # Load Lucene index using Pyserini
#     index_path = "/ivi/ilps/projects/TREC-Ikat-CW22/passage_index/trec_ikat_2023_passage_index"
#     print("Loading Lucene index...")
#     t0 = perf_counter()
#     searcher = LuceneSearcher(index_path)
#     print("Loading index took {:.3f} sec".format(perf_counter() - t0))


#     # Initialize SparseRetrieval once
#     evaluator = SparseRetrieval(config=config, model=model, dataset_name="ikat25",
#                                 compute_stats=True, dim_voc=model.output_dim)

#     print("Index loaded. Ready for interactive retrieval.")

#     # Load the cross-encoder model for reranking
#     cross_encoder_name = "cross-encoder/ms-marco-MiniLM-L6-v2"
#     tokenizer = AutoTokenizer.from_pretrained(cross_encoder_name)
#     cross_encoder = AutoModelForSequenceClassification.from_pretrained(cross_encoder_name)
#     cross_encoder.eval()
#     cross_encoder.to("cuda" if torch.cuda.is_available() else "cpu")

#     try:
#         while True:
#             # Interactive query input
#             user_query = input("\n\nEnter your query (or type 'exit' to quit):\n").strip()
#             if user_query.lower() == "exit":
#                 print("Exiting interactive retrieval.")
#                 break

#             # Create a temporary query collection
#             temp_query_path = "/tmp/temp_query.tsv"
#             with open(temp_query_path, "w") as f:
#                 f.write(f"0\t{user_query}\n")

#             # Load the query into a DataLoader
#             q_collection = CollectionDatasetPreLoad(data_dir=temp_query_path, id_style="row_id")
#             q_loader = CollectionDataLoader(dataset=q_collection, tokenizer_type=model_training_config["tokenizer_type"],
#                                             max_length=model_training_config["max_length"], batch_size=1,
#                                             shuffle=False, num_workers=1)

#             # Perform retrieval
#             top_k_results = evaluator.retrieve(q_loader, top_k=exp_dict["config"]["top_k"], threshold=exp_dict["config"]["threshold"], return_d=True)

#             # Rerank the top 100 passages
#             rerank_top_k(user_query, top_k_results["retrieval"], tokenizer, cross_encoder, searcher)

#     except KeyboardInterrupt:
#         print("\nInteractive session terminated by user.")

#     # Cleanup
#     evaluator = None
#     gc.collect()
#     torch.cuda.empty_cache()


# def rerank_top_k(query, top_k_results, tokenizer, cross_encoder, searcher):
#     """
#     Rerank the top 100 passages using the cross-encoder model.
#     """
#     for query_id, passage_scores in top_k_results.items():
#         # Extract the top 100 passage IDs based on their scores
#         top_100_passages = sorted(passage_scores.items(), key=lambda x: x[1], reverse=True)[:100]
#         passage_ids = [passage_id for passage_id, _ in top_100_passages]

#         # Retrieve passage text using Pyserini's batch_doc
#         t0 = perf_counter()
#         passage_text_mapping = searcher.batch_doc(passage_ids, threads=128)
#         print("Retrieving passage text took {:.3f} sec".format(perf_counter() - t0))

#         # Create query-passage pairs
#         query_passage_pairs = []
#         for passage_id in passage_ids:
#             raw_passage = passage_text_mapping.get(passage_id)
#             if raw_passage:
#                 passage_text = json.loads(raw_passage.raw())['contents']
#                 query_passage_pairs.append(f"{query} [SEP] {passage_text}")

#         # Prepare query-passage pairs
#         inputs = tokenizer(
#             query_passage_pairs,
#             padding=True,
#             truncation=True,
#             return_tensors="pt",
#             max_length=512
#         )
#         inputs = {key: val.to("cuda" if torch.cuda.is_available() else "cpu") for key, val in inputs.items()}

#         # Perform inference
#         with torch.no_grad():
#             scores = cross_encoder(**inputs).logits.squeeze(-1).cpu().tolist()

#         # Attach scores to passages and sort
#         passages = [
#             {"id": passage_id, "text": json.loads(passage_text_mapping[passage_id].raw())['contents'], "rerank_score": scores[i]}
#             for i, passage_id in enumerate(passage_ids)
#         ]
#         passages.sort(key=lambda x: x["rerank_score"], reverse=True)

#         # Print the reranked results
#         print(f"\nQuery: {query}")
#         for i, passage in enumerate(passages[:10]):  # Print top 10 reranked results
#             print("--------------------")
#             print("Passage ID:", passage["id"])
#             print(f"Rank {i + 1}: (Score: {passage['rerank_score']})\n{passage['text']}\n")


# if __name__ == "__main__":
#     retrieve_evaluate()


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
    index_path = "/ivi/ilps/projects/TREC-Ikat-CW22/passage_index/trec_ikat_2023_passage_index"
    print("Loading Lucene index...")
    t0 = perf_counter()
    searcher = LuceneSearcher(index_path)
    print("Loading index took {:.3f} sec".format(perf_counter() - t0))

    # Initialize SparseRetrieval once
    evaluator = SparseRetrieval(config=config, model=model, dataset_name="ikat25",
                                compute_stats=True, dim_voc=model.output_dim)

    print("Index loaded. Ready for interactive retrieval.")

    # Load the cross-encoder model for reranking
    cross_encoder_name = "naver/trecdl22-crossencoder-debertav3"
    tokenizer = AutoTokenizer.from_pretrained(cross_encoder_name)
    cross_encoder = AutoModelForSequenceClassification.from_pretrained(cross_encoder_name)
    cross_encoder.eval()
    cross_encoder.to("cuda" if torch.cuda.is_available() else "cpu")

    try:
        while True:
            # Interactive query input
            user_query = input("\n\nEnter your query (or type 'exit' to quit):\n").strip()
            if user_query.lower() == "exit":
                print("Exiting interactive retrieval.")
                break

            # Create a temporary query collection
            temp_query_path = "/tmp/temp_query.tsv"
            with open(temp_query_path, "w") as f:
                f.write(f"0\t{user_query}\n")

            # Load the query into a DataLoader
            q_collection = CollectionDatasetPreLoad(data_dir=temp_query_path, id_style="row_id")
            q_loader = CollectionDataLoader(dataset=q_collection, tokenizer_type=model_training_config["tokenizer_type"],
                                            max_length=model_training_config["max_length"], batch_size=1,
                                            shuffle=False, num_workers=1)

            # Perform retrieval
            top_k_results = evaluator.retrieve(q_loader, top_k=exp_dict["config"]["top_k"], threshold=exp_dict["config"]["threshold"], return_d=True)

            # Rerank the top 100 passages
            rerank_top_k(user_query, top_k_results["retrieval"], tokenizer, cross_encoder, searcher)

    except KeyboardInterrupt:
        print("\nInteractive session terminated by user.")

    # Cleanup
    evaluator = None
    gc.collect()
    torch.cuda.empty_cache()


def rerank_top_k(query, top_k_results, tokenizer, cross_encoder, searcher):
    """
    Rerank the top 100 passages using the cross-encoder model.
    """
    for query_id, passage_scores in top_k_results.items():
        # Extract the top 100 passage IDs based on their scores
        top_100_passages = sorted(passage_scores.items(), key=lambda x: x[1], reverse=True)[:100]
        passage_ids = [passage_id for passage_id, _ in top_100_passages]

        # Retrieve passage text using Pyserini's batch_doc
        t0 = perf_counter()
        passage_text_mapping = searcher.batch_doc(passage_ids, threads=128)
        print("Retrieving passage text took {:.3f} sec".format(perf_counter() - t0))

        # Create query-passage pairs
        query_passage_pairs = []
        for passage_id in passage_ids:
            raw_passage = passage_text_mapping.get(passage_id)
            if raw_passage:
                passage_text = json.loads(raw_passage.raw())['contents']
                query_passage_pairs.append((query, passage_text))

        # Tokenize and rerank
        scores = []
        for query_text, passage_text in tqdm(query_passage_pairs, desc="Reranking passages"):
            inputs = tokenizer(query_text, passage_text, return_tensors="pt", truncation=True, padding=True).to(cross_encoder.device)
            with torch.no_grad():
                outputs = cross_encoder(**inputs)
                scores.append(outputs.logits[0].item())

        # Attach scores to passages and sort
        passages = [
            {"id": passage_id, "text": json.loads(passage_text_mapping[passage_id].raw())['contents'], "rerank_score": scores[i]}
            for i, passage_id in enumerate(passage_ids)
        ]
        passages.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Print the reranked results
        print(f"\nQuery: {query}")
        for i, passage in enumerate(passages[:20]):  # Print top 10 reranked results
            print("--------------------")
            print("Passage ID:", passage["id"])
            print(f"Rank {i + 1}: (Score: {passage['rerank_score']})\n{passage['text']}\n")


if __name__ == "__main__":
    retrieve_evaluate()