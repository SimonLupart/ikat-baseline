from openai import OpenAI
import json
from tqdm import tqdm
import os

from parse_utils import parse_raw

API_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(
  api_key=API_key
)

def rewrite_gpt(data, output_file):
    prompt_string = """# Instruction:
I will give you a conversation between a user and a system. Also, I will give you some background about the user. You should rewrite the last question of the user into a self-contained query.

# Background knowledge:
{ptkb}
# Context:
{ctx}
# Please rewrite the following user question:
{user_utterance}
# Re-written query: 
"""

    with open(output_file, "w") as tsv_queries_raw_4o:

        for turn_id, ctx_user in tqdm(data.items()):

            ptkb=ctx_user["ptkb"]
            ctx = ctx_user["context"]
            user_utterance=ctx_user["utterance"]

            rewrite = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt_string.format(ptkb=ptkb, ctx=ctx, user_utterance=user_utterance)}
            ],
            )

            rewrite = rewrite.choices[0].message.content
            tsv_queries_raw_4o.write(turn_id+"\t"+rewrite+"\n")


data=json.load(open("DATA/2023_flatten_test_topics.json"))
rewrite_gpt(data, "DATA/queries_gpt4o_2023.tsv")

data=json.load(open("DATA/2024_flatten_test_topics.json"))
rewrite_gpt(data, "DATA/queries_gpt4o_2024.tsv")
