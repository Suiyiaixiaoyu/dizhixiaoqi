from transformers import BertModel,BertPreTrainedModel
import torch.nn as nn
from model.resnet import ResModel
import torch
import torch.nn.functional as F

class bertmodel(BertPreTrainedModel):
    def __init__(self,config):
        super(bertmodel, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.2)

    def forward(self,qk_input_ids,qk_attention_mask):
        output = self.bert(qk_input_ids,qk_attention_mask)
        # cls = torch.mul(qk_attention_mask.unsqueeze(-1),output[0])
        cls = self.dropout(output[0])
        return cls

class model(nn.Module):
    def __init__(self,args,bertmodel):
        super(model, self).__init__()
        self.enconder = bertmodel
        self.c_embed = nn.Embedding(3,1024)
        self.conv2d = nn.Conv2d(1, 3, (2 * 5 + 1, 3), padding=(5, 0))
        # self.res = ResModel()
        self.dense = nn.Linear(9*1024,3)
        self.pool = nn.AdaptiveMaxPool2d((9,1024))
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.GELU()

    def forward(self,qk_input_ids,qk_attention_mask):
        batch_size = qk_attention_mask.shape[0]
        labels = self.c_embed.weight.data
        c = labels.transpose(-2,-1)
        c = F.normalize(c,p=2,dim=0)
        cls = self.enconder(qk_input_ids,qk_attention_mask)
        cls1 = torch.mul(qk_attention_mask.unsqueeze(-1), cls)
        cls1 = F.normalize(cls1,p=2,dim=2)
        g = torch.matmul(cls1,c)
        g = g.unsqueeze(1)
        u = F.relu(self.conv2d(g))
        umax,_ = torch.max(u.squeeze(-1), dim=1)
        attention_score = torch.softmax(umax,dim=-1)
        attention_score = attention_score.unsqueeze(-1)
        z = torch.mul(cls, attention_score)
        # out = self.res(z.unsqueeze(1))
        pred = self.dense(self.dropout(self.relu(self.pool(z).reshape(batch_size,-1))))

        return pred


