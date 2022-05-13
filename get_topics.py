from ast import arg
from email import parser
from email.policy import default
from bertopic import BERTopic
from argparse import ArgumentParserr
import os
import pandas as pd
from tqdm import tqdm

parser = ArgumentParserr()
parser.add_argument("--dir", default="", type=str, help="Directory path to saved model.")

args = parser.parse_args()

def main():
    topic_model = BERTopic.load(os.path.join(args.dir, "topic_model"))

    topic_list = []

    for i in tqdm(range(250)):
        topic = topic_model.get_topic(i)
        topic_list.append(
            " ".join([j for j, k in topic])
        )

    df = pd.DataFrame(
        {
            "topic": topic_list
        }
    )

    df.to_csv(os.path.join(args.dir, "topic_map.csv"), index=False)


if __name__ == "__main__":
    main()