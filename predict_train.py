import json
import json
from tqdm import tqdm
import os
import numpy as np
from transformers import BertTokenizerFast,AdamW
import torch
from tools.argsparsetool import getparse
from tools.dataprocess import loaddata,datasplit,set_seed
from tools.dealdata import EmotionDataset,PadBatchSeq
from model.xqmodel import model,bertmodel
from torch.utils.data import DataLoader
from tools.evaltool import evalfun
from adv.adversarial import PGD

args = getparse().parse_args()
#选择device
device = torch.device(f'cuda:{args.GPUNUM}') if torch.cuda.is_available() else torch.device('cpu')
set_seed(26)
tokenizer = BertTokenizerFast.from_pretrained(args.bert_path)
test = loaddata(args.test_path)
model = torch.load('xq_model.bin')
model = model.to(device)

#加载数据
data = loaddata(args.data_path)
train,dev = datasplit(data,args.num_train)

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
            pred = torch.softmax(pred,dim=1)[0]
            pred_index = torch.argmax(pred,dim=0)
            if pred[pred_index] > 0.9 and pred_index == 0:
                data['candidate'][index]['label'] = '完全匹配'
            elif pred[pred_index] > 0.9 and pred_index == 1:
                data['candidate'][index]['label'] = '部分匹配'
            elif pred[pred_index] > 0.9 and pred_index == 2:
                data['candidate'][index]['label'] = '不匹配'

        xx = []
        for index,x in enumerate(data['candidate']):
            if 'label' in x:
                xx.append(data['candidate'][index])

        data['candidate'] = xx

        if data['candidate'] != []:
            train.append(data)



train_dataset = EmotionDataset(train,tokenizer,args.max_length,istrain=1)
dev_dataset = EmotionDataset(dev,tokenizer,args.max_length,istrain=0)
train_loader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,collate_fn=PadBatchSeq(tokenizer.pad_token_id))
dev_loader = DataLoader(dev_dataset,batch_size=args.batch_size,shuffle=False,collate_fn=PadBatchSeq(tokenizer.pad_token_id))


pgd = PGD(model,emb_name='word_embeddings.',epsilon=0.8,alpha=0.2)
optim = AdamW(model.parameters(),lr=args.lr)
loss_fun = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([0.5,0.3,0.2]),size_average=True)
loss_fun = loss_fun.to(device)
m_f = 0
K = 3
#训练模型
def train_fun(m_f):
    train_loss = 0
    pbar = tqdm(train_loader)
    model.train()
    for step,batch in enumerate(pbar):
        qk_input_ids = batch['qk_input_ids'].long().to(device)
        qk_attention_mask = batch['qk_attention_mask'].to(device)
        label = batch['label'].to(device)

        pred = model(qk_input_ids,qk_attention_mask)


        loss = loss_fun(pred,label)

        train_loss +=loss.item()

        loss.backward()
        pgd.backup_grad()
        for t in range(K):
            pgd.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.data
            if t != K-1:
                model.zero_grad()
            else:
                pgd.restore_grad()
            pred_adv = model(qk_input_ids,qk_attention_mask)
            loss_adv = loss_fun(pred_adv,label)
            loss_adv.backward()
        pgd.restore()
        optim.step()
        optim.zero_grad()
        pbar.update()
        pbar.set_description(f'train_loss:{train_loss}')

        if step % 1000==0 and step != 0:
            model.eval()
            num_corr_1 = 0
            num_corr_2 = 0
            num_corr_3 = 0
            predg1,predg2,predg3 = 0,0,0
            goldg1,goldg2,goldg3 = 0,0,0
            with torch.no_grad():
                for batch in dev_loader:
                    qk_input_ids = batch['qk_input_ids'].long().to(device)
                    qk_attention_mask = batch['qk_attention_mask'].to(device)

                    label = batch['label'].to(device)
                    goldg1 += torch.sum(label==0).cpu()
                    goldg2 += torch.sum(label==1).cpu()
                    goldg3 += torch.sum(label==2).cpu()

                    pred = model(qk_input_ids,qk_attention_mask)
                    predg1 += torch.sum(torch.argmax(pred,dim=-1)==0).cpu().numpy()
                    predg2 += torch.sum(torch.argmax(pred,dim=-1)==1).cpu().numpy()
                    predg3 += torch.sum(torch.argmax(pred,dim=-1)==2).cpu().numpy()

                    num_corr_1 += len(np.intersect1d((torch.argmax(pred, dim=-1) == 0).nonzero().squeeze(-1).cpu().numpy(),
                                                     (label == 0).nonzero().squeeze(-1).cpu().numpy()))
                    num_corr_2 += len(np.intersect1d((torch.argmax(pred, dim=-1) == 1).nonzero().squeeze(-1).cpu().numpy(),
                                                     (label == 1).nonzero().squeeze(-1).cpu().numpy()))
                    num_corr_3 += len(np.intersect1d((torch.argmax(pred, dim=-1) == 2).nonzero().squeeze(-1).cpu().numpy(),
                                                     (label == 2).nonzero().squeeze(-1).cpu().numpy()))



            f1 = evalfun(num_corr_1, predg1, goldg1)
            f2 = evalfun(num_corr_2, predg2, goldg2)
            f3 = evalfun(num_corr_3, predg3, goldg3)

            f = (f1+f2+f3)/3
            print('f1:',f)
            if m_f < f:
                m_f = f
                print('model save')
                torch.save(model,'xq_new_model.bin')
            model.train()
    return m_f

for epoch in range(6):
    m_f = train_fun(m_f)
    torch.save(model, 'new_model.bin')

print(m_f)