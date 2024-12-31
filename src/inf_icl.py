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
import json

import nltk
import re
import fire
import time

CITE_TOKEN = "[URL_CITE]"
RES_DIR = "/data/group1/z40436a/ME/URL_Citation_Classification_Intermediate/result/output"

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

class Command:
    def __init__(self, icl_method:str, seed:int, model_name:str):
        '''
        icl_method: random, bm25, encoder
        seed: [111, 5374, 93279]
        model_name: ["meta-llama/Llama-3.1-8B-Instruct", ]
        '''
        self.icl_method = icl_method
        self.seed = int(seed)
        self.model_name = model_name
    
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

def read_icl(file_path:str) -> list[list[int]]:
    '''
    return icl_idx top-k (from left)
    '''
    icl_idxs = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line == '\n':
                break
            icl_idxs.append(json.loads(line))
    return icl_idxs

def create_inst(train_df:pd.DataFrame, test_df:pd.DataFrame, icl_ids:list[list[int]], k:int=5) -> list[list[dict[str, str]]]:
    texts = []
    
    train_replaced_sentences = replace_tag(train_df['citation-paragraph'])
    train_conts = get_3sent(train_replaced_sentences)

    test_replaced_sentences = replace_tag(test_df['citation-paragraph'])
    test_conts = get_3sent(test_replaced_sentences)

    for test_cont, (i, row) in zip(test_conts, test_df.iterrows()):
        reset_idx = 0
        instruction = [
            {"role":"System", "content": f"""Your task is to classify the type of artifact (TYPE) reffered to the URL and the citation reason (FUNCTION). I will provide you with a URL and citation context, section titles.\n
Here is the classification schema for the artifact type:
1. Tool: toolkit, software, system
2. Code: codebase, library, API
3. Dataset: corpus, image, sets
4. Knowledge: lexicon, knowledge graph
5. DataSource: source data for the Dataset/Knowledge
6. Document: specifications, guidelines
7. Paper: scholarly papers
8. Media: games, music, videos
9. Website: services, homepages
10. Mixed: citations referring to multiple resources
    
Here is the classification schema for the citation reason:
1. Use: Used in the citing paper’s research
2. Produce: First produced or released by the citing paper’s research
3. Compare: Compared with other resources
4. Extend: Used in the citing paper’s research but are improved, upgraded, or changed during the research
5. Introduce: The resources or the related information
6. Other: The URL citation does not belong to the above categories"""}
        ]

        if k == 0:
            pass
        elif k > 0 and k <=5:
            for top_k in range(k):
                icl_idx = icl_ids[reset_idx][top_k]
                icl_df = train_df.iloc[icl_idx]
                # print(icl_df)
                icl_input = f"""Please classify the artifact type and the citation reason for the following URL and citation sentence.
URL: {icl_df['url']}
Citation Context: {train_conts[icl_idx]}
Footnote or Reference text (if exists): {icl_df['citation-info']}
Section Titles (if exists): {icl_df['passage-title']}"""
                instruction.append({"role":"user", "content": icl_input})
                instruction.append({"role":"assistant", "content": f"""TYPE: {icl_df['type']}\nFUNCTION: {row['function'].split("（")[0]}"""})
        else:
            print("error")

        test_input = f"""Please classify the artifact type and the citation reason for the following URL and citation sentence.
URL: {row['url']}
Citation Context: {test_cont}
Footnote or Reference text (if exists): {row['citation-info']}
Section Titles (if exists): {row['passage-title']}"""
        instruction.append({"role":"user", "content": test_input})

        reset_idx += 1

        texts.append(instruction)
    return texts


def main(c:Command) -> None:
    csv_dataset = pd.read_csv("/data/group1/z40436a/ME/URL_Citation_Classification_Intermediate/data/all_data.csv", encoding="utf-8")
    
    train_df, eval_df = train_test_split(csv_dataset, test_size = 0.1, random_state=int(c.seed))
    print("train_data_size:::", len(train_df))
    print("test_data_size:::", len(eval_df))

    icl_idxs = read_icl(f"/data/group1/z40436a/ME/URL_Citation_Classification_Intermediate/icl/{c.icl_method}/{str(c.seed)}.txt")

    pipe = pipeline(
            "text-generation",
            model=c.model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
            max_new_tokens = 50
        )

    # measure time
    time_logs = []

    for k in range(1,5+1):
        print(f"###{k}shot")
        eval_texts = create_inst(train_df, eval_df, icl_idxs, k)

        ### REWRITE ME!!!
        prompts = eval_texts
        # with open('./log.txt', 'a') as log_file:
        #     log_file.write("===eval_text===")
        #     log_file.write(str(prompts))

        # start time
        start_time = time.time()
        
        with torch.no_grad():
            outputs = pipe(
                prompts,
                do_sample=False
            )

        # end time
        end_time = time.time()
        # elapse time
        elapsed_time = end_time - start_time

        # log
        # with open('./log.txt', 'a') as log_file:
        #     log_file.write("===output===")
        #     log_file.write(str(outputs))

        with open(f"{RES_DIR}/{c.model_name}/{c.icl_method}/{str(c.seed)}_{str(k)}shot.json", "w") as output_file:
            # json.dump(generated_texts, output_file, indent=4)
            json.dump(outputs, output_file, indent=4)

        # append time log
        time_logs.append({"k-shot": k, "elapsed_time": elapsed_time})

    with open(f"{RES_DIR}/{c.model_name}/{c.icl_method}/{str(c.seed)}_time_log.json", "w") as time_log_file:
        json.dump(time_logs, time_log_file, indent=4)

if __name__ == "__main__":
    c = fire.Fire(Command)
    main(c)