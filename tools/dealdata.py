from torch.utils.data import Dataset
import torch
import re

class EmotionDataset(Dataset):
    def __init__(self,datas,tok,max_lengths,istrain):
        super(EmotionDataset, self).__init__()
        self.max_lengths = max_lengths
        self.data = EmotionDataset.makedataset(datas, tok, max_lengths,istrain)

    @staticmethod
    def makedataset(datas,tok,maxlengths,istrain):
        qks_input_ids = []
        qks_attention_mask = []
        # ks_input_ids = []
        # ks_attention_mask = []
        labels = []
        if istrain:
            for data in datas:
                query = data['query']
                query = tok(query[:maxlengths],return_tensors='pt')
                q_input_ids = query.input_ids[0]
                q_attention_mask = query.attention_mask[0]
                qks_input_ids.append(torch.cat([q_input_ids,q_input_ids[1:]],dim=0))
                qks_attention_mask.append(torch.cat([q_attention_mask,q_attention_mask[1:]],dim=0))
                labels.append(0)
                for candidate in data['candidate']:
                    key = tok(candidate['text'][:maxlengths], return_tensors='pt')
                    k_input_ids = key.input_ids[0]
                    k_attention_mask = key.attention_mask[0]
                    qks_input_ids.append(torch.cat([q_input_ids, k_input_ids[1:]], dim=0))
                    qks_attention_mask.append(torch.cat([q_attention_mask, k_attention_mask[1:]], dim=0))
                    if candidate['label'] == '不匹配' :
                        labels.append(2)
                    elif candidate['label'] == '部分匹配':
                        labels.append(1)
                    elif candidate['label'] == '完全匹配':
                        labels.append(0)
            dataset = list(zip(qks_input_ids,qks_attention_mask,labels))
        else:
            for data in datas:
                query = data['query']
                query = tok(query[:maxlengths], return_tensors='pt')
                q_input_ids = query.input_ids[0]
                q_attention_mask = query.attention_mask[0]
                for candidate in data['candidate']:
                    key = tok(candidate['text'][:maxlengths], return_tensors='pt')
                    k_input_ids = key.input_ids[0]
                    k_attention_mask = key.attention_mask[0]
                    qks_input_ids.append(torch.cat([q_input_ids, k_input_ids[1:]], dim=0))
                    qks_attention_mask.append(torch.cat([q_attention_mask, k_attention_mask[1:]], dim=0))
                    if candidate['label'] == '不匹配':
                        labels.append(2)
                    elif candidate['label'] == '部分匹配':
                        labels.append(1)
                    elif candidate['label'] == '完全匹配':
                        labels.append(0)

            dataset = list(zip(qks_input_ids, qks_attention_mask, labels))


        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        qk_input_ids, qk_attention_mask,label = self.data[idx]
        return {'qk_input_ids':qk_input_ids, 'qk_attention_mask':qk_attention_mask,'label':label}

class PadBatchSeq:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        res = dict()
        res['label'] = torch.LongTensor([i['label'] for i in batch])
        max_len = max([len(i['qk_input_ids']) for i in batch])
        res['qk_input_ids'] = torch.LongTensor([i['qk_input_ids'].numpy().tolist() + [self.pad_id] * (max_len - len(i['qk_input_ids'])) for i in batch])
        res['qk_attention_mask'] = torch.LongTensor([i['qk_attention_mask'].numpy().tolist() + [self.pad_id] * (max_len - len(i['qk_attention_mask'])) for i in batch])

        return res










