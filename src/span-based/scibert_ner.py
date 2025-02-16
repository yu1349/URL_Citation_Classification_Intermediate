import spacy
import random
import json
from spacy_llm.util import assemble
from spacy.tokens import DocBin
from spacy.util import filter_spans
import os
from tqdm import tqdm

# CONST
ANNO_IDX = 435
EX_NAME = 'jsai'
DATA_DIR = f"../../data/{EX_NAME}_data_split"

def read_split_ids(ids_path:str) -> list[int]:
    '''
    データ分割を読む関数
    - 引数
    ids_path: データ分割のインデックスが格納されたファイルパス
    - 返り値
    データ分割のインデックスが格納されたリスト
    '''
    with open(ids_path, 'r', encoding='utf-8') as ids_file:
        ids_lst = ids_file.readlines()
    ids_lst = [int(id) for id in ids_lst]
    return ids_lst

def make_pre_data(pre_data:list[dict]) -> list:
    '''
    spacyに渡す前の整形を行う関数
    - 引数
    pre_data: jsonファイル
    - 返り値
    json
    
    '''
    urls = []
    ret_data = []
    for example in pre_data['examples']:
        temp_dict = {}
        temp_dict['text'] = example['content']
        temp_dict['entities'] = []
        for annotation in example['annotations']:
            start = annotation['start']
            end = annotation['end']
            label = annotation['tag_name'].upper().replace(' ', '')
            temp_dict['entities'].append((start, end, label))

            if label == 'URL':
                url = temp_dict['text'][start:end]
        urls.append(url)
        ret_data.append(temp_dict)
    return urls, ret_data

def make_spacy_data(urls, full_pre_data, split_name):
    # split_idを取得
    split_ids = read_split_ids(f'{DATA_DIR}/{split_name}_ids.txt')
    # split_dataを取得
    split_data = [full_pre_data[split_id] for split_id in split_ids]
    split_url = [urls[split_id] for split_id in split_ids]

    nlp = spacy.blank('en')
    doc_bin = DocBin()
    loop_id = 0
    for training_example in tqdm(split_data):
        text = training_example['text']
        labels = training_example['entities']
        doc = nlp.make_doc(text)
        ents = []
        for start, end, label in labels:
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                print(span, file=open(f'{DATA_DIR}/skipping_data.txt', 'a', encoding='utf-8'))
            else:
                ents.append(span)
        filtered_ents = filter_spans(ents)
        doc.ents = filtered_ents
        doc.user_data['URL'] = split_url[loop_id]
        # print(doc.user_data)
        doc_bin.add(doc)

        loop_id += 1

    # print(span)
    doc_bin.to_disk(f"./data/{split_name}.spacy")
    return doc_bin

def make_examples(docs):
    # スペシファイされた形式に変換
    formatted_examples = []

    for doc in docs:
        example = {"text": doc.text, "spans": []}
        
        for ent in doc.ents:
            span_info = {
                "text": ent.text,
                "is_entity": True,  # ここではすべてのエンティティを 'True' として設定
                "label": ent.label_,
                "reason": "N/A"  # 理由は後で追加するか、特定のロジックを使って設定
            }
            example["spans"].append(span_info)

        # エンティティが含まれていない部分（フレーバーなど）を追加
        # 例として、"chocolate" をエンティティとして `is_entity=False` にする場合
        # 必要に応じて `reason` を記述できます
        # for token in doc:
        #     if token.text.lower() not in [ent.text.lower() for ent in doc.ents]:  # すでにエンティティでないトークンを調べる
        #         span_info = {
        #             "text": token.text,
        #             "is_entity": False,
        #             "label": "==NONE==",
        #         }
        #         example["spans"].append(span_info)
        
        formatted_examples.append(example)
    return formatted_examples

# ICLの分割IDを読み込む
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

def main()->None:
    # モデルの読み込み
    nlp = spacy.load('/data/group1/z40436a/ME/URL_Citation_Classification_Intermediate/src/span-based/model/allenai/scibert_scivocab_uncased/model-best')

    template = """
You are an expert Named Entity Recognition (NER) system.
Your task is to accept Text as input and extract metadata entities about research data referred to in the URL.
Entities must have one of the following labels: {{ ', '.join(labels) }}.
If a span is not an entity label it: `==NONE==`.
{# whitespace #}
{# whitespace #}
{%- if description -%}
{# whitespace #}
{{ description }}
{# whitespace #}
{%- endif -%}
{%- if label_definitions -%}
Below are definitions of each label to help aid you in what kinds of named entities to extract for each label.
Assume these definitions are written by an expert and follow them closely.
{# whitespace #}
{%- for label, definition in label_definitions.items() -%}
{{ label }}: {{ definition }}
{# whitespace #}
{%- endfor -%}
{# whitespace #}
{# whitespace #}
{%- endif -%}
{%- if prompt_examples -%}
Q: Given the paragraph below, identify a list of entities, and for each entry explain why it is or is not an entity:
{# whitespace #}
{# whitespace #}
{%- for example in prompt_examples -%}
Paragraph: {{ example.text }}
Answer:
{# whitespace #}
{%- for span in example.spans -%}
{{ loop.index }}. {{ span.to_str() }}
{# whitespace #}
{%- endfor -%}
{# whitespace #}
{# whitespace #}
{%- endfor -%}
{%- else -%}
{# whitespace #}
Here is an example of the output format for a paragraph using different labels than this task requires.
Only use this output format but use the labels provided
above instead of the ones defined in the example below.
Do not output anything besides entities in this output format.
Output entities in the order they occur in the input paragraph regardless of label.

Q: Given the paragraph below, identify a list of entities, and for each entry explain why it is or is not an entity:

Paragraph: Sriracha sauce goes really well with hoisin stir fry, but you should add it after you use the wok.
Answer:
1. Sriracha sauce | True | INGREDIENT | is an ingredient to add to a stir fry
2. really well | False | ==NONE== | is a description of how well sriracha sauce goes with hoisin stir fry
3. hoisin stir fry | True | DISH | is a dish with stir fry vegetables and hoisin sauce
4. wok | True | EQUIPMENT | is a piece of cooking equipment used to stir fry ingredients
{# whitespace #}
{# whitespace #}
{%- endif -%}
Paragraph: {{ text }}
Answer:"""

    # nlp.config["components"]["llm"]["task"]["template"] = str(template)
    # データを呼び出し
    INPUT_DIR = './data'
    train_doc_bin = DocBin().from_disk(f'{INPUT_DIR}/train.spacy')
    train_docs = list(train_doc_bin.get_docs(nlp.vocab))
    dev_doc_bin = DocBin().from_disk(f'{INPUT_DIR}/dev.spacy')
    dev_docs = list(dev_doc_bin.get_docs(nlp.vocab))    # イテレーターに変換

    random.seed(0)

    train_examples = make_examples(train_docs)
    train_icls = read_icl('./icl/for_dev/random.txt')
    
    for k in range(0, 0+1): # REWRITE ME!!!
        loop_id = 0
        print(f"---k=scibert---")
        outputs = []
        for dev_doc in tqdm(dev_docs):
            # print(dev_doc.user_data)
            if k > 0:
                icls = [train_icls[loop_id][i] for i in range(k)]
                # print(icls)
                examples = [train_examples[i] for i in icls]
                # print(nlp.config["components"]["llm"]["task"]["examples"])
                nlp.config["components"]["llm"]["task"]["examples"] = examples
                # print(len(nlp.config["components"]["llm"]["task"]["examples"]), nlp.config["components"]["llm"]["task"]["examples"])
            else:
                pass

            # 出力
            # nlp.to_disk('./tmp_config')
            # nlp = assemble('./tmp_config/config.cfg')
            # print("test_text:::", dev_doc.text)
            output = nlp(dev_doc.text)
            outputs.append(output)

            # インクリメント
            loop_id += 1
        results = [{'text': output.text, 'entities': [{'start': ent.start_char, 'end': ent.end_char, 'label': ent.label_} for ent in output.ents]} for output in outputs]
        
        # ファイルに出力
        output_dir = f'./res/allenai/scibert_scivocab_uncased' 
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/{str(k)}.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        # break

# テスト
if __name__ == "__main__":
    main()
    # result = infer(test_text)
    # print("Entities:", result)
