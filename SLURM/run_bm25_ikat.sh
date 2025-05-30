#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=adhoc
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=32gb #120gb
#SBATCH -c 8
#SBATCH --output=/ivi/ilps/projects/ikat24/essir24-convsearch/SLURM/out/adhoc-%j.out

# Set-up the environment.
source /home/slupart/.bashrc
conda activate ikat24
nvidia-smi


# python -m bm25.retrieve --index_path /ivi/ilps/projects/TREC-Ikat-CW22/passage_index/trec_ikat_2023_passage_index \
#                         --topics DATA/queries_manual_2024.tsv \
#                         --output EXP/manual_bm25/IKAT2024/run.json

# python -m bm25.retrieve --index_path /ivi/ilps/projects/TREC-Ikat-CW22/passage_index/trec_ikat_2023_passage_index \
#                         --topics DATA/queries_gpt4o_2024.tsv \
#                         --output EXP/gpt4o_bm25/IKAT2024/run.json

python -m bm25.retrieve --index_path /ivi/ilps/projects/TREC-Ikat-CW22/passage_index/trec_ikat_2023_passage_index \
                        --topics DATA/queries_manual_2023.tsv \
                        --output EXP/manual_bm25/IKAT2023/run.json

# python -m bm25.retrieve --index_path /ivi/ilps/projects/TREC-Ikat-CW22/passage_index/trec_ikat_2023_passage_index \
#                         --topics DATA/queries_gpt4o_2023.tsv \
#                         --output EXP/gpt4o_bm25/IKAT2023/run.json