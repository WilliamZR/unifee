from base_templates import BasicModule
from transformers import AutoModel,AutoConfig
import torch
import torch.nn as nn

#using the cls of claim
class RobertaCls(BasicModule):
    def __init__(self, args):
        super(RobertaCls, self).__init__()
        self.config = AutoConfig.from_pretrained(args.bert_name, num_labels=len(args.id2label))
        hidden_size = self.config.hidden_size
        self.linear1 = nn.Linear(hidden_size, 128)
        self.relu = nn.ReLU()
        # self.linear2 = nn.Linear(128, len(args.id2label))
        self.linear2 = nn.Linear(128, 1)

        self.init_weights()
        self.args = args

        self.bert = AutoModel.from_pretrained(args.bert_name, config=self.config)
        self.dropout = nn.Dropout(args.dropout)

        self.count_parameters()
        # self.print_modules()

    def forward(self, batch, args, test_mode):
        pos_ids, pos_attention_mask, neg_ids, neg_attention_mask = batch
        pos_out = self.bert(pos_ids, attention_mask=pos_attention_mask)[1]

        if test_mode:
            output = pos_out
        else:
            neg_out = self.bert(neg_ids, attention_mask = neg_attention_mask)[1]
            output = torch.cat([pos_out, neg_out], dim=0)

        hg = self.dropout(output)
        hg = self.relu(self.linear1(hg))
        hg = self.linear2(hg)

        if test_mode:
            pos_scores = torch.sigmoid(hg).view(-1)
            neg_scores = None
        else:
            scores = torch.sigmoid(hg).view(2,-1)
            pos_scores = scores[0]
            neg_scores = scores[1]

        golds = torch.ones_like(pos_scores)

        return pos_scores, neg_scores, golds