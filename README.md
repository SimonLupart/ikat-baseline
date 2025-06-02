# Neural Conversational Search Baselines for TREC iKAT

This repository provide baselines and tools for the **TREC iKAT** track: the *Interactive Knowledge Assistance Track* at TREC. This track focuses on **neural conversational search** with user personalization, context, and goal-driven information seeking.


The TREC iKAT track models **goal-oriented conversations** where a user interacts with a search agent to complete a complex task (e.g., finding a university, preparing for a trip, obtaining a hunting license). The agent can use both dialogue history and user profile information (PTKB) to personalize and adapt its retrieval. See [guidelines](https://www.trecikat.com/guidelines/) for more details.

#### We provide in this repository:

* SPLADE-based retrieval [[SPLADE Retrieval]](#splade-retrieval)
* Query rewriting with PTKB integration (manual and GPT-based)
* BM25 baselines [[BM25 Retrieval]](#bm25-retrieval)
* Cross-encoder reranking (MiniLM and DeBERTa) [[Reranking]](#reranking)
* Evaluation of the runs [[Results]](#results)


#### Example of information seeking conversation:

> **PTKB**: *I live in the Netherlands. I have a bachelor’s degree in computer science from Tilburg University.*

**[User]**: `I want to start my master’s degree, can you help me with finding a university?`  
**[System]**: *Do you want to study abroad?*  
**[User]**: `No, I don't want to go abroad.`  
**[System]**: *I can help you find a university to continue your studies in the Netherlands as a computer science student. Here are the Top Computer Science Universities in the Netherlands*:
1. *Delft University of Technology*
2. *Eindhoven University of Technology*
3. *Vrije Universiteit Amsterdam*

**[User]**: `...`

## Installation

### Conda Environment

```bash
conda env create -f environment.yml
conda activate ikat24
```

### Data

Download the iKAT dataset:

```bash
bash ./DATA/dl_ikat.sh
```

Also required:
* SPLADE NumBa index (see DATA/README.md)
* Pyserini Lucene index (see DATA/README.md)

## SPLADE retrieval

> Requires 256GB RAM and one GPU.

Running SPLADE retrieval on human manual rewritten queries (both iKAT 2023 and iKAT 2024):

```bash
export SPLADE_CONFIG_NAME="config_hf_splade_ikat.yaml"

index_dir=path/to/splade_index
eval_queries=[DATA/queries_manual_2023.tsv,DATA/queries_manual_2024.tsv]
out_dir=EXP/manual_splade

python -m splade.retrieve init_dict.model_type_or_dir=naver/splade-cocondenser-ensembledistil \
        config.pretrained_no_yamlconfig=true config.index_dir=$index_dir \
        config.out_dir=$out_dir \
        data.Q_COLLECTION_PATH=$eval_queries 
```

Now using a GPT4o-mini rewrite to integrate the PTKB and generate the rewrite:

```bash
export OPENAI_API_KEY="your_openai_key"

python DATA/rewrite_gpt_ikat.py
```

Now running SPLADE on the GPT40-mini rewrite:

```bash
export SPLADE_CONFIG_NAME="config_hf_splade_ikat.yaml"

index_dir=path/to/splade_index
eval_queries=[DATA/queries_gpt4o_2023.tsv,DATA/queries_gpt4o_2024.tsv]
out_dir=EXP/gpt4o_splade

python -m splade.retrieve init_dict.model_type_or_dir=naver/splade-cocondenser-ensembledistil \
        config.pretrained_no_yamlconfig=true config.index_dir=$index_dir \
        config.out_dir=$out_dir \
        data.Q_COLLECTION_PATH=$eval_queries
```

## BM25 retrieval

Similarly you can do retrieval with BM25, this requires less ressource.

With gpt4o-mini rewrite (on iKAT 2024):
```bash
python -m bm25.retrieve --index_path path/to/pyserini_index \
                        --topics DATA/queries_gpt4o_2024.tsv \
                        --output EXP/gpt4o_bm25/IKAT2024/run.json
```

You can evaluate the runs with:
> For evaluate we use the binary qrels for recall and mrr, and the graded relevance for ndcg.

```bash
python -m splade.evaluate --run_dir EXP/gpt4o_bm25/IKAT2024/run.json \
                          --qrel_file_path DATA/2024_test_qrels.json \
                          --qrel_binary_file_path DATA/2024_test_qrels_binary.json
```

## Reranking

For reranking we also use the pyserini index, to load the text from the retrieved documents id.

```bash
python -m rerank.rerank --index_path path/to/pyserini_index \
                        --model naver/trecdl22-crossencoder-debertav3 \
                        --run EXP/gpt4o_splade/IKAT2024/run.json \
                        --query_file DATA/queries_gpt4o_2024.tsv \
                        --output EXP/gpt4o_splade_rerank/IKAT2024/run.json

python -m rerank.rerank --index_path path/to/pyserini_index \
                        --model naver/trecdl22-crossencoder-debertav3 \
                        --run EXP/gpt4o_bm25/IKAT2024/run.json \
                        --query_file DATA/queries_gpt4o_2024.tsv \
                        --output EXP/gpt4o_bm25_rerank/IKAT2024/run.json
```

You can use `naver/trecdl22-crossencoder-debertav3`, `cross-encoder/ms-marco-MiniLM-L-6-v2` or any HuggingFace models. This can also be run on iKAT 2023.

Similarly you can evaluate the produced runs:

```bash

python -m splade.evaluate --run_dir EXP/gpt4o_splade_rerank/IKAT2024/run.json \
                          --qrel_file_path DATA/2024_test_qrels.json \
                          --qrel_binary_file_path DATA/2024_test_qrels_binary.json

```
## Results

We provide here a brief summary of several produced runs:

### IKAT 2023

| Model     | Rerank             | Rewrite | nDCG@10 | Recall@100 | MRR@1000 |
|-----------|--------------------|---------|---------|------------|----------|
| BM25      | -                  | Human        |    0.2888      |      0.2941      |     0.3600     |
| SPLADE    | -                  | Human        |     0.2396     |      0.2855      |     0.3592     |
| BM25      | -                  | GPT4o-mini   |    0.1255      |      0.1441      |     0.1822     |
| SPLADE    | -                  | GPT4o-mini   |     0.1849     | 0.1975           |     0.3052     |

> Note that the pool of assessed passages in iKAT 2023 was biased toward BM25.

### IKAT 2024

| Model     | Rerank             | Rewrite | nDCG@10 | Recall@100 | MRR@1000 |
|-----------|--------------------|---------|---------|------------|----------|
| BM25      | -                  | Human        |    0.1944       |     0.2059       |    0.3617      |
| SPLADE    | -                  | Human        |     0.3401      |    0.3621        |    0.5703      |
| SPLADE    | -                  | GPT4o-mini   |     0.1849      | 0.1975           |   0.3052       |
| SPLADE    | MiniLM             | GPT4o-mini   |   0.3434        |     0.2954       |     0.5386     |
| SPLADE    | DeBERTa            | GPT4o-mini   |   0.3876        |     0.2954       |     0.5944     |

## SPLADE Interactive Retrieval

A iKAT searcher interactive tool.
> Need RAM

```bash
export SPLADE_CONFIG_NAME="config_hf_splade_ikat.yaml"

index_dir=/ivi/ilps/projects/ikat24/splade_index_website/splade_index/
out_dir=EXP/interact_splade
mkdir -p EXP/interact_splade

lucene_index=/ivi/ilps/projects/TREC-Ikat-CW22/passage_index/trec_ikat_2023_passage_index
temp_path_q=/tmp/temp_query.tsv

python -m splade.retrieve_interactive init_dict.model_type_or_dir=naver/splade-cocondenser-ensembledistil \
        config.pretrained_no_yamlconfig=true config.index_dir=$index_dir \
        config.out_dir=$out_dir \
        +config.lucene_index=$lucene_index \
        +config.temp_path_q=$temp_path_q
```

## Additionnal Resources

* Reproducibility Repository for ANCE dense retrieval on iKAT [[github]](https://github.com/EricLangezaal/PersonalizedCIR)

* TREC iKAT website: [https://www.trecikat.com/](https://www.trecikat.com/)
* Official TREC website: [https://trec.nist.gov/](https://trec.nist.gov/)

* Contact: [s.c.lupart@uva.nl](mailto:s.c.lupart@uva.nl)

## Citations

Please cite our TREC iKAT overview paper if you use this work:

```bibtex
@inproceedings{coordinators-trec2024-papers-proc-4,
    author = {Mohammad Aliannejadi (University of Amsterdam), Zahra Abbasiantaeb (University of Amsterdam), Simon Lupart (University of Amsterdam), Shubham Chatterjee (University of Edinburgh), Jeffrey Dalton (University of Edinburgh), Leif Azzopardi (University of Strathclyde)},
    title = {TREC iKAT 2024: The Interactive Knowledge Assistance Track Overview},
    booktitle = {The Thirty-Third Text REtrieval Conference Proceedings (TREC 2024), Gaithersburg, MD, USA, November 15-18, 2024},
    series = {NIST Special Publication},
    volume = {1329},
    publisher = {National Institute of Standards and Technology (NIST)},
    year = {2024},
    trec_org = {coordinators},
    trec_runs = {},
    trec_tracks = {ikat}
   url = {https://trec.nist.gov/pubs/trec33/papers/Overview_ikat.pdf}
}
```
