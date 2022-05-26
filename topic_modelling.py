from src.data.load_data import load_data_for_tm
from src.config.config import logger

from bertopic import BERTopic
from cuml.cluster import HDBSCAN
from cuml.manifold import UMAP
from argparse import ArgumentParser
import os
import time
import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm

import nltk
nltk.download('stopwords')
nltk.download('punkt')

import re

def clean_text(text):
    text = str(text)
    text = re.sub(r'[0-9"]', '', text) # number
    text = re.sub(r'#[\S]+\b', '', text) # hash
    text = re.sub(r'@[\S]+\b', '', text) # mention
    text = re.sub(r'https?\S+', '', text) # link
    text = re.sub(r'\s+', ' ', text) # multiple white spaces

    return text

parser = ArgumentParser()

parser.add_argument("--output_dir", type=str, default="")
parser.add_argument("--num_topics", type=int, default=250)

args = parser.parse_args()

def main():
    st = time.time()
    logger.info("Loading data from mongo db...")
    df = load_data_for_tm()
    logger.info("Data Loaded with {} samples (took {} seconds).".format(len(df), time.time() - st))

    text_list = df.body.tolist()

    prep_text_list = []    
    stop_words = set(stopwords.words('english'))

    for sent in tqdm(text_list):
        word_tokens = word_tokenize(sent)
        filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
        prep_sent = clean_text(" ".join(filtered_sentence))
        # print(sent, "\n", prep_sent)
        prep_text_list.append(prep_sent)

    df['preprocessed'] = prep_text_list

    st = time.time()
    logger.info("Initializing model...")
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric='cosine',
        low_memory=False
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=10,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )
    topic_model = BERTopic(
        language="english", 
        verbose=True, 
        umap_model=umap_model, 
        hdbscan_model=hdbscan_model,
        calculate_probabilities=True,
        nr_topics = args.num_topics
    )
    logger.info("Model initialized (took {} seconds).".format(time.time() - st))
    
    st = time.time()
    logger.info("Fitting model...")
    topics, probs = topic_model.fit_transform(prep_text_list)
    logger.info("Model fit successful (took {} seconds).".format(time.time() - st))

    st = time.time()
    logger.info("Saving Model to {}...".format(args.output_dir))
    topic_model.save(os.path.join(args.output_dir, "topic_model"))
    logger.info("Model saved (took {} seconds).".format(time.time() - st))
    
    st = time.time()
    logger.info("Saving Probabilities to {}...".format(args.output_dir))
    np.save(os.path.join(args.output_dir, "probabilities.npy"), probs)
    logger.info("Probabilities saved (took {} seconds).".format(time.time() - st))

    st = time.time()
    logger.info("Saving output to {}...".format(args.output_dir))
    df["topic"] = topics
    df = df[["_id", "created_at", "author", "topic"]]
    df.to_csv(os.path.join(args.output_dir, "modeled_thoughts.csv"), index=False)
    logger.info("Output saved (took {} seconds).".format(time.time() - st))
    logger.info("Done!")

if __name__=="__main__":
    main()