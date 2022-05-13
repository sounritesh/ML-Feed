from src.data.load_data import load_data_for_tm
from src.config.config import logger

from bertopic import BERTopic
from argparse import ArgumentParser
import os
import time

parser = ArgumentParser()

parser.add_argument("--output_dir", type=str, default="")
parser.add_argument("--num_topics", type=int, default=150)

args = parser.parse_args()

def main():
    st = time.time()
    logger.info("Loading data from mongo db...")
    df = load_data_for_tm()
    logger.info("Data Loaded with {} samples (took {} seconds).".format(len(df), time.time() - st))

    text_list = df.body.tolist()

    st = time.time()
    logger.info("Initializing model...")
    topic_model = BERTopic(language="english", verbose=True)
    logger.info("Model initialized (took {} seconds).".format(time.time() - st))
    
    st = time.time()
    logger.info("Fitting model...")
    topics, probs = topic_model.fit_transform(text_list)
    logger.info("Model fit successful (took {} seconds).".format(time.time() - st))

    st = time.time()
    logger.info("Saving Model to {}...".format(args.output_dir))
    topic_model.save(os.path.join(args.output_dir, "topic_model"))
    logger.info("Model saved (took {} seconds).".format(time.time() - st))
    
    st = time.time()
    logger.info("Saving output to {}...".format(args.output_dir))
    df["topic"] = topics
    df.to_csv(os.path.join(args.output_dir, "modeled_thoughts.csv"), index=False)
    logger.info("Output saved (took {} seconds).".format(time.time() - st))
    logger.info("Done!")

if __name__=="__main__":
    main()