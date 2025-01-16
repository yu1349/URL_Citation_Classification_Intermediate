import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import fire

GOLD_DATA_PATH = "/data/group1/z40436a/ME/URL_Citation_Classification_Intermediate/data/all_data.csv"
RES_DIR = "/data/group1/z40436a/ME/URL_Citation_Classification_Intermediate/result/output"

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

def pred_type_and_func(LLM_responses: json) -> tuple[list, list]:
    type_preds = []
    func_preds = []

    for each_response in LLM_responses:
        # print(each_response[-1]['generated_text'])
        completion = each_response[-1]['generated_text'][-1]
        if completion['role'] == 'assistant':
            try:
                # print("TYPE:", completion['content'].split("TYPE: ")[1].split()[0])
                type_preds.append(completion['content'].split("TYPE: ")[1].split()[0])
            except:
                print("format error")
                type_preds.append("FormatError")
            try:
                # print("FUNCTION:", completion['content'].split("FUNCTION: ")[1].split()[0])
                func_preds.append(completion['content'].split("FUNCTION: ")[1].split()[0])
            except:
                print("format error")
                func_preds.append("FormatError")

    return type_preds, func_preds

def evaluation_metrics(gold:list[str], pred:list[str]) -> None:
    acc = accuracy_score(gold, pred)
    pre = precision_score(gold, pred, average="macro")
    rec = recall_score(gold, pred, average="macro")
    f1 = f1_score(gold, pred, average="macro")

    return {"acc": acc, "pre":pre, "rec":rec, "f1":f1}

def main(c:Command):

    csv_dataset = pd.read_csv(GOLD_DATA_PATH, index_col=0)

    train_df, eval_df = train_test_split(csv_dataset, test_size = 0.1, random_state=int(c.seed))
    gold_types, gold_funcs = [], []
    for i, row in eval_df.iterrows():
        gold_types.append(row['type'])
        gold_funcs.append(row['function'].split("（")[0])

    for k in range(1, 5+1):
        pred_data_path = f"{RES_DIR}/{c.model_name}/{c.icl_method}/{str(c.seed)}_{str(k)}shot.json"
        
        with open(pred_data_path, 'r') as json_file:
            LLM_responses = json.load(json_file)

        pred_types, pred_funcs = pred_type_and_func(LLM_responses)

        metrics_types = evaluation_metrics(gold_types, pred_types)
        metrics_funcs = evaluation_metrics(gold_funcs, pred_funcs)

        with 



if __name__ == "__main__":
    c = fire.Fire(Command)
    main(c)