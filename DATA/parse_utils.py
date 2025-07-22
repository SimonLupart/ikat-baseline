import json
from tqdm import tqdm


import json

import json

def parse_raw(path_topics, year="2024"):
    """
    Store a flat version of the conversation for evaluating all turns.
    Store a tsv file with resolved utterance.
    Save both original and PTKB-augmented versions for 2025.
    """

    output_prefix = f"DATA/queries_manual_{year}"
    flattened_path = f"DATA/{year}_flatten_test_topics.json"
    flattened_path_extended = f"DATA/{year}_flatten_test_topics_extended.json"
    data = {}
    data_extended = {}

    with open(output_prefix + ".tsv", "w") as tsv_manual:
        with open(path_topics, "r") as tc:
            topics = json.load(tc)

        # Cache user utterances from -1 topics for reuse in -2
        utterance_cache = {}
        if year == "2025":
            for topic in topics:
                topic_number = str(topic["number"])
                if topic_number.endswith("-1"):
                    utterance_cache[topic_number] = [
                        turn["user_utterance"] for turn in topic["responses"]
                    ]

        for topic in topics:
            topic_number = str(topic["number"])
            context = ""

            if year == "2025":
                turns = topic["responses"]

                # --- original PTKB
                ptkb_raw = topic["ptkb"]
                ptkb_standard = ", ".join([f"{i+1}: {item}" for i, item in enumerate(ptkb_raw)])

                # --- extended PTKB (only if -2)
                if topic_number.endswith("-2"):
                    base_topic = topic_number.replace("-2", "-1")
                    user_utterances = utterance_cache.get(base_topic, [])
                    ptkb_extended = ", ".join([f"U{i+1}: {utt}" for i, utt in enumerate(user_utterances)])
                else:
                    ptkb_extended = ptkb_standard

            else:
                turns = topic["turns"]
                ptkb_raw = topic["ptkb"]
                ptkb_standard = ", ".join([f"{i+1}: {k}: {v}" for i, (k, v) in enumerate(ptkb_raw.items())])
                ptkb_extended = ptkb_standard  # same for non-2025

            for turn in turns:
                turn_id = str(turn["turn_id"])
                topic_turn_id = f"{topic_number}_{turn_id}"

                user_utterance = turn["user_utterance"] if year == "2025" else turn["utterance"]
                resolved_utterance = turn["resolved_utterance"]
                response = turn["response"]

                # Store original PTKB version
                data[topic_turn_id] = {
                    "ptkb": ptkb_standard,
                    "context": context,
                    "utterance": user_utterance
                }

                # Store extended PTKB version
                data_extended[topic_turn_id] = {
                    "ptkb": ptkb_extended,
                    "context": context,
                    "utterance": user_utterance
                }

                # Write resolved utterance
                tsv_manual.write(f"{topic_turn_id}\t{resolved_utterance}\n")

                # Update context
                context += f"\nUSER: {user_utterance}".strip()
                context += f"\nSYSTEM: {response}".strip()
                context = context.strip()

    # Write both versions
    with open(flattened_path, "w") as f_out:
        json.dump(data, f_out, indent=4)

    with open(flattened_path_extended, "w") as f_out_ext:
        json.dump(data_extended, f_out_ext, indent=4)




def trec2json(path_qrel, year="2024"):
    """
    Convert TREC qrel file to JSON format.
    """
    qrels = {}
    qrels_binary={}

    with open(path_qrel, 'r') as f:
        for line in f:
            qid, _, did, rel = line.strip().split()
            if qid not in qrels:
                qrels[qid] = {}
                qrels_binary[qid] = {}
            qrels[qid][did] = int(rel)
            if int(rel)>=2:
                qrels_binary[qid][did] = 1
            else:
                qrels_binary[qid][did] = 0

    json.dump(qrels, open("DATA/"+year+"_test_qrels.json", "w"), indent = 4)
    json.dump(qrels_binary, open("DATA/"+year+"_test_qrels_binary.json", "w"), indent = 4)

    return qrels



# if __name__ == "__main__":
# parse_raw("DATA/2023_test_topics.json", year="2023")
# parse_raw("DATA/2024_test_topics.json", year="2024")

# trec2json("DATA/2023_test_qrels.trec", year="2023")
# trec2json("DATA/2024_test_qrels.trec", year="2024")


parse_raw("DATA/2025_test_topics.json", year="2025")

