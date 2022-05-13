from pymongo import MongoClient
from tqdm import tqdm
import pandas as pd

from src.config.secrets import secrets

def load_data_for_tm():
    client = MongoClient(secrets["client"])

    thought_list = []
    thoughts = client.forum.thoughts
    for thought in tqdm(thoughts.find(no_cursor_timeout=True), total=thoughts.count_documents({})):
        is_top_level = 0 if 'c_parent' in thought and not not thought['c_parent'] else 1
        if is_top_level:
            thought_list.append(thought)
    print("{} thoughts retrieved from client.".format(len(thought_list)))
    df_thoughts = pd.DataFrame(thought_list)[["_id", "created_at","body", "author"]]

    return df_thoughts