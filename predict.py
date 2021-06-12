import json

from tools.dataprocess import loaddata
from tools.argsparsetool import getparse
from transformers import BertTokenizerFast
import torch
from tqdm import tqdm

args = getparse().parse_args()
device = torch.device(f'cuda:{args.GPUNUM}') if torch.cuda.is_available() else torch.device('cpu')
tokenizer = BertTokenizerFast.from_pretrained(args.bert_path)
test = loaddata(args.test_path)
model = torch.load('new_model.bin')
model = model.to(device)

preds = []
with torch.no_grad():
    model.eval()
    pbar = tqdm(test)
    for data in pbar:
        query = data['query']
        q_input_ids = tokenizer(query,return_tensors='pt')['input_ids'][0]
        q_attention_mask = tokenizer(query, return_tensors='pt')['attention_mask'][0]
        candidate = data['candidate']
        for index,key in enumerate(candidate):
            text = key['text']
            k_input_ids = tokenizer(text,return_tensors='pt')['input_ids'][0]
            k_attention_mask = tokenizer(text,return_tensors='pt')['attention_mask'][0]
            input_ids = torch.cat([q_input_ids, k_input_ids[1:]], dim=0).to(device).unsqueeze(0)
            attention_mask = torch.cat([q_attention_mask, k_attention_mask[1:]], dim=0).to(device).unsqueeze(0)
            pred = model(input_ids,attention_mask)
            pred = torch.argmax(pred,dim=-1).cpu().data.item()
            if pred == 0:
                data['candidate'][index]['label'] = '完全匹配'
            elif pred == 1:
                data['candidate'][index]['label'] = '部分匹配'
            elif pred == 2:
                data['candidate'][index]['label'] = '不匹配'
        preds.append(data)
f = open('柯哀无敌_ addr_match_runid.txt', 'w', encoding='utf-8')
for pred in preds:
    f.write(json.dumps(pred,ensure_ascii=False)+"\n")

f.close()







