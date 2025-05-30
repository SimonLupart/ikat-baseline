#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=adhoc
#SBATCH --partition gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-01:00:00
#SBATCH --mem=120gb #120gb
#SBATCH -c 4
#SBATCH --output=/projects/0/prjs0871/hackathon/conv-search/EXP/log_slurm/adhoc-%j.out

# Set-up the environment.
conda activate cs_ikat
nvidia-smi

# Inference regular
index_dir=/ivi/ilps/projects/ikat24/splade_index_website/splade_index/
eval_queries=[TBD.tsv]

python -m splade.retrieve init_dict.model_type_or_dir=naver/splade-cocondenser-ensembledistil \
        config.pretrained_no_yamlconfig=true config.index_dir="$index_dir" \
        config.out_dir="/ivi/ilps/projects/ikat24/essir24-convsearch/EXP/adhoc/out" \
        data.Q_COLLECTION_PATH=[$eval_queries]


python -m bm25.retrieve --index_path /ivi/ilps/projects/TREC-Ikat-CW22/passage_index/trec_ikat_2023_passage_index \
                        --topics DATA/queries_manual_2024.tsv \
                        --output EXP/manual_bm25/IKAT2024/run.json

python -m bm25.retrieve --index_path /ivi/ilps/projects/TREC-Ikat-CW22/passage_index/trec_ikat_2023_passage_index \
                        --topics DATA/queries_gpt4o_2024.tsv \
                        --output EXP/gpt4o_bm25/IKAT2024/run.json

python -m bm25.retrieve --index_path /ivi/ilps/projects/TREC-Ikat-CW22/passage_index/trec_ikat_2023_passage_index \
                        --topics DATA/queries_manual_2023.tsv \
                        --output EXP/manual_bm25/IKAT2023/run.json

python -m bm25.retrieve --index_path /ivi/ilps/projects/TREC-Ikat-CW22/passage_index/trec_ikat_2023_passage_index \
                        --topics DATA/queries_gpt4o_2023.tsv \
                        --output EXP/gpt4o_bm25/IKAT2023/run.json