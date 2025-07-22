# 2023 topics
# https://drive.google.com/file/d/1zPSiAqLmbx9QFGm6walnuMUl7xoJmRB7/view
gdown -O DATA/2023_test_topics.json https://drive.google.com/uc?id=1zPSiAqLmbx9QFGm6walnuMUl7xoJmRB7
# 2023 qrels
wget -O DATA/2023_test_qrels.trec https://trec.nist.gov/data/ikat/2023-qrels.all-turns.txt

# 2024 topics
# https://drive.google.com/file/d/1pIu_kSn6YHQRX74DFknmjUJA34-JY67M/view
gdown -O DATA/2024_test_topics.json https://drive.google.com/uc?id=1pIu_kSn6YHQRX74DFknmjUJA34-JY67M
# 2024 qrels
wget -O DATA/2024_test_qrels.trec  https://trec.nist.gov/data/ikat/2024-qrels.txt

# 2025 topics
wget -O DATA/2025_test_topics.json https://raw.githubusercontent.com/irlabamsterdam/iKAT/main/2025/data/2025_test_topics.json
# no qrel for 2025 yet

python DATA/parse_utils.py
