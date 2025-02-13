 # -- coding: utf-8 --
import transformers
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import json
import torch
from transformers.modeling_outputs import ModelOutput
import torch.nn as nn 
from transformers import EvalPrediction, AutoTokenizer, AutoModel, TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import os
from fire import Fire



# ラベルのマッピング
## str->int
ROLE_MAP = {'Method': 0,
            'Material': 1,
            }

TYPE_MAP = {'Tool': 0,
            'Code': 1,
            'Dataset': 2,
            'Knowledge': 3,
            'DataSource': 4
            }

FUNCTION_MAP = {'Use': 0,
                'Produce': 1,
                'Compare': 2,
                'Extend': 3,
                'Introduce': 4,
                'Other': 5
                }

# こちらで設定したトークン
## tokenizerに渡して、1 tokenとして扱う
SPECIAL_TOKENS = [f'[Cite_Footnote_{str(i)}]' for i in range(0, 30)] + ['[SEP]']

class Model_Path_Init():
    def __init__(self, model_path='allenai/scibert_scivocab_uncased'):
        self.model_path = model_path    # allenai/scibert_scivocab_uncased, google-bert/bert-base-uncased


# データセットの整形
class Mydatasets(Dataset):
    def __init__(self, X: list[str], Y: list[list[int]], n_label=1) -> None:
        self.X = X
        self.Y = Y  # (n_label, datanum)
        self.n_label = n_label

        self.datanum = len(self.X)

    def __len__(self) -> int:
        return self.datanum

    def __getitem__(self, idx: int) -> tuple[str, list[int]]:
        return self.X[idx], [self.Y[i][idx] for i in range(self.n_label)]

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


def make_dataset(dataset:pd.DataFrame, n_label:int) -> Mydatasets:
    '''
    Trainerに渡すためのdatasetを作成する関数
    - 引数
    dataset: 分割後のデータセット（Dataframe）
    n_label: 分類層の数（マルチタスク学習）
    - 返り値
    Trainerに渡すためのデータセット（Mydatasets）
    '''
    res_dataset = Mydatasets(
        dataset.apply(lambda row: f"{row['url']}[SEP]{row['section_title']}[SEP]{row['text']}" if row['add_info'] is None    # 参考文献や脚注がある場合、[SEP]で繋ぐ
                      else f"{row['url']}[SEP]{row['section_title']}[SEP]{row['text']}[SEP]{row['add_info']}[SEP]{row['add_info']}",
                      axis=1),
        [dataset['role'].map(lambda x: ROLE_MAP.get(x)),    # ここでマッピングを行う
         dataset['type'].map(lambda x: TYPE_MAP.get(x)), 
         dataset['func'].map(lambda x: FUNCTION_MAP.get(x))],
        n_label
    )
    return res_dataset

def load_tokenizer(model_name: str, special_tokens: list[str]) -> transformers.AutoTokenizer:
    '''
    tokenizerを読み込む関数
    - 引数
    model_name: huggingfaceのモデルパス
    special_tokens: tokenとして扱いたい特殊な文字列（[Cite_Footnote]など）
    - 返り値
    トークナイザー
    '''
    if 'deberta' in model_name:
        tokenizer = transformers.DebertaV2Tokenizer.from_pretrained(model_name)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_tokens(special_tokens, special_tokens=True)

    return tokenizer

class Multitask_Model(nn.Module):
    '''
    マルチタスク学習を行うためのモデル
    - 引数
    vocab_size: 使用するトークナイザーの語彙サイズ
    n_classes: 分類層の数[それぞれのクラス数]
    model_name: huggingfaceのモデルパス
    - 基本構造
    中間層: BERT
    出力層: CLSトークンに独立した分類層をかませる
    Loss: それぞれの分類層でのCEの和
    '''
    def __init__(self, vocab_size: int, n_classes: list[int], model_name) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.bert.resize_token_embeddings(vocab_size)

        self.n_classes = n_classes

        self.dropout = torch.nn.Dropout(0.1)
        self.hidden_dim = self.bert.config.hidden_size

        self.classifiers = torch.nn.ModuleList([
            torch.nn.Linear(self.hidden_dim, n_class) for n_class in n_classes
        ])
        
    def forward(self,
                input_ids,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                label=None,):
        # use only CLS token of last layer
        X = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids
        ).last_hidden_state[:, 0, :]
        X = self.dropout(X)

        logits = []
        for classifier in self.classifiers:
            logits.append(classifier(X))

        loss = None
        losses = []
        if label is not None:
            assert len(label) == len(self.n_classes)
            for i, logit in enumerate(logits):
                losses.append(
                    torch.nn.functional.cross_entropy(logit, label[i]))
            loss = sum(losses)

        return ModelOutput(loss=loss, logits=logits)

# データ収集
class DataCollator():
    def __init__(self, tokenizer, max_length=512) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        x, y = list(zip(*batch))
        n_label = len(y[-1])
        labels = [torch.tensor([i[j] for i in y]).long()
                  for j in range(n_label)]

        inputs = self.tokenizer(
            x,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        inputs['label'] = labels

        return inputs

def compute_metrics(res: EvalPrediction):
    n_label = len(res.label_ids)
    labels = res.label_ids
    # check if the classification is binary or not
    if res.predictions[0].ndim == 1:
        preds = [res.predictions[i] > 0.5 for i in range(n_label)]
        accs = [accuracy_score(labels[i], preds[i]) for i in range(n_label)]
        recalls = [recall_score(labels[i], preds[i], average='binary')
                   for i in range(n_label)]
        precisions = [precision_score(
            labels[i], preds[i], average='binary') for i in range(n_label)]
        f1s = [f1_score(labels[i], preds[i], average='binary')
               for i in range(n_label)]
        avr_acc = np.mean(accs)  # type: ignore
        avr_recall = np.mean(recalls)  # type: ignore
        avr_precision = np.mean(precisions)  # type: ignore
        avr_f1 = np.mean(f1s)  # type: ignore
        ret = {f'label{i}_acc': accs[i] for i in range(n_label)}
        ret.update({f'label{i}_recall': recalls[i] for i in range(n_label)})
        ret.update(
            {f'label{i}_precision': precisions[i] for i in range(n_label)})
        ret.update({f'label{i}_f1': f1s[i] for i in range(n_label)})
        ret.update({
            'avr_acc': avr_acc,
            'avr_recall': avr_recall,
            'avr_precision': avr_precision,
            'avr_f1': avr_f1
        })
        return ret

    preds = [res.predictions[i].argmax(axis=1) for i in range(n_label)]

    accs = [accuracy_score(labels[i], preds[i]) for i in range(n_label)]
    macro_recalls = [recall_score(
        labels[i], preds[i], average='macro') for i in range(n_label)]
    micro_recalls = [recall_score(
        labels[i], preds[i], average='micro') for i in range(n_label)]
    macro_precisions = [precision_score(
        labels[i], preds[i], average='macro') for i in range(n_label)]
    micro_precisions = [precision_score(
        labels[i], preds[i], average='micro') for i in range(n_label)]
    macro_f1s = [f1_score(labels[i], preds[i], average='macro')
                 for i in range(n_label)]
    micro_f1s = [f1_score(labels[i], preds[i], average='micro')
                 for i in range(n_label)]
    weighted_f1s = [f1_score(labels[i], preds[i], average='weighted')
                    for i in range(n_label)]

    avr_acc = np.mean(accs)  # type: ignore
    avr_macro_recall = np.mean(macro_recalls)  # type: ignore
    avr_micro_recall = np.mean(micro_recalls)  # type: ignore
    avr_macro_precision = np.mean(macro_precisions)  # type: ignore
    avr_micro_precision = np.mean(micro_precisions)  # type: ignore
    avr_macro_f1 = np.mean(macro_f1s)  # type: ignore
    avr_micro_f1 = np.mean(micro_f1s)  # type: ignore

    ret = {f'label{i}_acc': accs[i] for i in range(n_label)}
    ret.update(
        {f'label{i}_macro_recall': macro_recalls[i] for i in range(n_label)})
    ret.update(
        {f'label{i}_micro_recall': micro_recalls[i] for i in range(n_label)})
    ret.update(
        {f'label{i}_macro_precision': macro_precisions[i] for i in range(n_label)})
    ret.update(
        {f'label{i}_micro_precision': micro_precisions[i] for i in range(n_label)})
    ret.update({f'label{i}_macro_f1': macro_f1s[i] for i in range(n_label)})
    ret.update({f'label{i}_micro_f1': micro_f1s[i] for i in range(n_label)})
    ret.update(
        {f'label{i}_weighted_f1': weighted_f1s[i] for i in range(n_label)})
    ret.update({'avr_acc': avr_acc,
                'avr_macro_recall': avr_macro_recall,
                'avr_micro_recall': avr_micro_recall,
                'avr_macro_precision': avr_macro_precision,
                'avr_micro_precision': avr_micro_precision,
                'avr_macro_f1': avr_macro_f1,
                'avr_micro_f1': avr_micro_f1
                })

    return ret

def main(c)->None:
    MODEL_NAME = c.model_path
    # 定数の定義
    EX_NAME = 'full'
    INPUT_DATA_PATH = f'./data/{EX_NAME}_data_split/input_data.json'
    SPLIT_DIR = f'./data/{EX_NAME}_data_split'
    MODEL_LOG_DIR = f'./log/{EX_NAME}/{MODEL_NAME}'
    MODEL_SAVE_DIR = f'./model/{EX_NAME}/{MODEL_NAME}'
    RES_DIR = f'./res/{EX_NAME}/{MODEL_NAME}'

    CONST_FILE_PATH = [INPUT_DATA_PATH, SPLIT_DIR, MODEL_LOG_DIR, MODEL_SAVE_DIR, RES_DIR]
    for path in CONST_FILE_PATH:
        print(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)

    # TEST CODE
    # return 0

    # 前処理をしたデータのロード
    with open(INPUT_DATA_PATH, 'r', encoding='utf-8') as json_file:
        input_data = json.load(json_file)

    # それぞれの分割に対応するデータの準備
    train_ids = read_split_ids(f'{SPLIT_DIR}/train_ids.txt')
    dev_ids = read_split_ids(f'{SPLIT_DIR}/dev_ids.txt')
    test_ids = read_split_ids(f'{SPLIT_DIR}/test_ids.txt')
    train_data = pd.DataFrame([input_data[train_id] for train_id in train_ids])
    dev_data = pd.DataFrame([input_data[dev_id] for dev_id in dev_ids])
    test_data = pd.DataFrame([input_data[test_id] for test_id in test_ids])

    # 分類層とそのラベル数の定義
    n_classes = [2, 5, 6]

    # 学習のシードの決定
    transformers.set_seed(0)

    # トークナイザーの初期化
    tokenizer = load_tokenizer(MODEL_NAME, SPECIAL_TOKENS)

    # Trainerに渡すデータセットの作成
    train_dataset = make_dataset(train_data, len(n_classes))
    dev_dataset = make_dataset(dev_data, len(n_classes))
    test_dataset = make_dataset(test_data, len(n_classes))
    collator = DataCollator(tokenizer)

    # TEST CODE
    # return 0

    # モデルの初期化
    ## モデルをGPUに乗っける
    model = Multitask_Model(
            len(tokenizer),
            n_classes,
            model_name=MODEL_NAME
        ).cuda()

    # GPUの連続値にモデルを乗っける
    for param in model.parameters(): 
        param.data = param.data.contiguous()
    
    # パラメータ等の設定
    training_args = {
            'logging_dir': MODEL_LOG_DIR,
            'output_dir': MODEL_SAVE_DIR,
            'evaluation_strategy': 'epoch',
            'logging_strategy': 'epoch',
            'save_strategy': 'epoch',
            'save_total_limit': 1,
            'overwrite_output_dir': True,
            'fp16': True,
            'label_names': ['label'],
            'lr_scheduler_type': 'constant',
            # 'lr_scheduler_type': 'cosine',
            # 'metric_for_best_model': 'avr_macro_f1',
            'metric_for_best_model': 'loss',
            'load_best_model_at_end': True,
            'per_device_train_batch_size': 8,
            'per_device_eval_batch_size': 16,
            # 'per_device_train_batch_size': 32,
            # 'per_device_eval_batch_size': 64,
            # 'warmup_steps': 1000,
            'num_train_epochs': 20,  # REWRITE ME!!!
            'remove_unused_columns': False,
            'report_to': 'all',
            'learning_rate': 1e-5,
        }
    
    training_args2 = TrainingArguments(
        **training_args
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        args=training_args2,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    # 学習の実行
    trainer.train(ignore_keys_for_eval=['input_ids', 'attention_mask', 'token_type_ids'])
    trainer.save_model(MODEL_SAVE_DIR)
    # save trainer
    trainer.save_state()

    # ここから結果の書き出し
    train_pred = trainer.evaluate(train_dataset, ignore_keys=['loss'])
    os.makedirs(os.path.dirname(f'{RES_DIR}/{str(training_args['num_train_epochs'])}/train.json'), exist_ok=True)
    json.dump(train_pred, open(f'{RES_DIR}/{str(training_args['num_train_epochs'])}/train.json', 'w'))

    dev_pred = trainer.evaluate(dev_dataset, ignore_keys=['loss'])
    os.makedirs(os.path.dirname(f'{RES_DIR}/{str(training_args['num_train_epochs'])}/dev.json'), exist_ok=True)
    json.dump(dev_pred, open(f'{RES_DIR}/{str(training_args['num_train_epochs'])}/dev.json', 'w'))

    test_pred = trainer.evaluate(test_dataset, ignore_keys=['loss'])
    os.makedirs(os.path.dirname(f'{RES_DIR}/{str(training_args['num_train_epochs'])}/test.json'), exist_ok=True)
    json.dump(test_pred, open(f'{RES_DIR}/{str(training_args['num_train_epochs'])}/test.json', 'w'))

if __name__ == "__main__":
    c = Fire(Model_Path_Init)
    main(c)