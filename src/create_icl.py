import json
import os
import torch
from datasets import load_dataset, Dataset
import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)

from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
import random
import numpy as np
from typing import Literal, Optional, TypedDict
from sklearn.model_selection import train_test_split

import nltk
import re

CITE_TOKEN = "[URL_CITE]"

class URLCiteDataset(torch.utils.data.Dataset):
    '''
    create dataset
    - init
    - len
    - getitem
    '''
    def __init__(self, texts: list[str]):
        self.texts = texts

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx]
    
def replace_tag(sentences: pd.Series) -> list[str]:
    # replace [Cite_****] to [Cite] token
    rule = re.compile(r'\[Cite[^\[\] ]*\]')
    sentences_replaced:list[str] = list()
    for sentence in sentences:
        sentences_replaced.append(rule.sub(CITE_TOKEN, sentence))

    return sentences_replaced

def get_3sent(paragraphs:list[str]) -> list[str]:
    ret:list[list[str]] = list()
    for paragraph in paragraphs:
        sentences: list[str] = nltk.sent_tokenize(paragraph)
        if not len(sentences):
            print('!!!')
        if len(sentences) < 4:
            ret.append(sentences)
            continue
        else:
            for i in range(len(sentences)):
                if CITE_TOKEN in sentences[i]:
                    if i == 0:
                        ret.append(sentences[i:i+2])
                    elif i == len(sentences)-1:
                        ret.append(sentences[i-1:i+1])
                    else:
                        ret.append(sentences[i-1:i+2])
                    break
                if i == len(sentences)-1:
                    # print(sentences)
                    pass
    cont_3sent = [" ".join(sent) for sent in ret]
    return cont_3sent

def create_icl(train_df:pd.DataFrame, test_df:pd.DataFrame, method:str, k:int=5) -> list[list[str]]:
    icl_idxs: list[list[str]] = []

    train_replaced_sentences = replace_tag(train_df['citation-paragraph'])
    test_replaced_sentences = replace_tag(test_df['citation-paragraph'])

    # check
    print(len(train_replaced_sentences))
    print(len(test_replaced_sentences))

    train_cont_3sent = get_3sent(train_replaced_sentences)
    test_cont_3sent = get_3sent(test_replaced_sentences)

    # check
    print(len(train_cont_3sent))
    print(len(test_cont_3sent))

    # random
    if method == "random":
        for cont, (i, row) in zip(test_cont_3sent, test_df.iterrows()):
            random.seed(i)
            icl_idxs.append(random.sample(range(len(train_cont_3sent)), k))

    #bm25
    elif method == "bm25":
        tokenized_corpus = [cont.split(" ") for cont in train_cont_3sent]
        bm25 = BM25Okapi(tokenized_corpus)

        for cont, (i, row) in zip(test_cont_3sent, test_df.iterrows()):
            bm25_scores = bm25.get_scores(cont.split(" "))
            icl_idxs.append(np.argsort(bm25_scores)[-k:][::-1].tolist())
    elif method == "encoder":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SentenceTransformer("intfloat/multilingual-e5-base").to(device)

        tokenized_corpus = model.encode(train_cont_3sent, convert_to_tensor=True, device=device)
        for cont, (i, row) in zip(test_cont_3sent, test_df.iterrows()):
            tokenized_query = model.encode(cont, convert_to_tensor=True, device=device)
            cos_scores = util.cos_sim(tokenized_query, tokenized_corpus)[0].cpu().numpy()

            icl_idxs.append(np.argsort(-cos_scores)[:k].tolist())
    else:
        print("select other method")
    
    return icl_idxs

def main():
    csv_dataset = pd.read_csv("/data/group1/z40436a/ME/URL_Citation_Classification_Intermediate/data/all_data.csv", encoding="utf-8")

    seeds = [111, 5374, 93279]
    for seed in seeds:
        train_df, eval_df = train_test_split(csv_dataset, test_size = 0.1, random_state=seed)
        print("train_data_size:::", len(train_df))
        print("test_data_size:::", len(eval_df))

        methods = ["random", "bm25", "encoder"]
        # methods = ["encoder"]
        for method in methods:
            icls = create_icl(train_df, eval_df, method=method)

            output_dir = f"/data/group1/z40436a/ME/URL_Citation_Classification_Intermediate/icl/{method}"
            with open(f"{output_dir}/{str(seed)}.txt", "w") as jsonl_file:
                for icl in icls:
                    json.dump(icl, jsonl_file)
                    jsonl_file.write("\n")

if __name__ == "__main__":
    main()