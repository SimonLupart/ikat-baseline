import json
from tqdm import tqdm


def parse_raw(path_topics, year="2024"):
    """
    Store a flat version of the conversation for evaluating all turns.
    Store a tsv file with resolved utterance.
    """

    with open("DATA/queries_manual_"+year+".tsv", "w") as tsv_manual:
        data={}
        with open(path_topics,"r") as tc:
            topics = json.load(tc)
            for topic in topics:
                topic_number = str(topic["number"])
                context=""
                ptkb_raw = topic["ptkb"]
                ptkb=""
                for k,v in ptkb_raw.items():
                    ptkb+=str(k)+ ": " + str(v) + ", "
                ptkb = ptkb[:-2] # remove additional ", " from the end

                for index, turn in enumerate(topic['turns']):
                    # id
                    turn_id = str(turn['turn_id'])
                    topic_turn_id = str(topic_number) + "_" + str(turn_id)
                    # store the current utterance
                    user_utterance=turn['utterance']
                    data[topic_turn_id]={"ptkb":ptkb,"context":context,"utterance":user_utterance}
                    # write the human resolved utterance
                    user_utterance_manual=turn['resolved_utterance']
                    tsv_manual.write(str(topic_turn_id)+"\t"+user_utterance_manual+"\n")
                    # update context
                    context=context+"\nUSER: "+turn['utterance']
                    context = context.strip()
                    context=context+"\nSYSTEM: "+turn["response"]
                    context = context.strip()

    json.dump(data, open("DATA/"+year+"_flatten_test_topics.json", "w"), indent = 4)


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
parse_raw("DATA/2023_test_topics.json", year="2023")
parse_raw("DATA/2024_test_topics.json", year="2024")

trec2json("DATA/2023_test_qrels.trec", year="2023")
trec2json("DATA/2024_test_qrels.trec", year="2024")