from transformers import AutoModel,AutoConfig
from base_templates import BasicModule
import torch
import torch.nn as nn
from dgl.nn.pytorch.conv import GATConv, GraphConv

class FusionEvidenceModel(BasicModule):
    def __init__(self, args):
        super(FusionEvidenceModel, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)
        num_headers = 1

        self.sent_bert = AutoModel.from_pretrained(args.roberta_path)
        sent_hidden_size = 768
        fusion_graph_hidden_size = sent_hidden_size
        self.sent_linear = nn.Linear(sent_hidden_size, fusion_graph_hidden_size)
        self.sent_relu = nn.ReLU()
        self.fusion_gnn = GATConv(fusion_graph_hidden_size, fusion_graph_hidden_size, num_headers, residual= True)
        self.sent_linear1 = nn.Linear(fusion_graph_hidden_size, 128)
        self.sent_linear2 = nn.Linear(128, 2)

        # table
        self.table_bert = AutoModel.from_pretrained(args.tapas_path)
        table_hidden_size = 768

        # segment
        self.segment_relu = nn.ReLU()
        self.segment_linear = nn.Linear(table_hidden_size, fusion_graph_hidden_size)
        self.segment_linear1 = nn.Linear(fusion_graph_hidden_size, 128)
        self.segment_linear2 = nn.Linear(128, 2)

        # cell
        cell_graph_hidden_size = table_hidden_size
        self.cell_relu = nn.ReLU()
        
        self.cell_linear = nn.Linear(table_hidden_size, cell_graph_hidden_size)
        self.cell_gnn = GATConv(cell_graph_hidden_size, cell_graph_hidden_size, num_headers, residual= True)
        self.cell_linear1 = nn.Linear(fusion_graph_hidden_size, 128)
        self.cell_linear2 = nn.Linear(128, 2)

       #self.init_weights()
        # self.count_parameters()

        # self.bert = AutoModel.from_pretrained(args.bert_name, config=self.config)
        # self.gnn = GATConv(hidden_size, hidden_size, num_headers)

        self.count_parameters()
        # self.print_modules()

    def forward(self, batch, args):
        # [batch_size * num_sentences, seq_len]
        sent_input_ids, sent_input_mask\
            , table_input_ids, table_input_mask, table_token_type_ids\
            , fusion_graph, cell_graph\
            , segment_pooling_matrix, cell_pooling_matrix = batch


        # [num_sentences in batched graph, seq_len]
        # initialize nodes of sentences
        sent_outputs = self.sent_bert(sent_input_ids, attention_mask = sent_input_mask)[1]
        sent_embs = self.sent_relu(self.sent_linear(sent_outputs))
        num_sentences = sent_embs.size()[0]
        # [# of tables, hiddden_size]
        table_outputs = self.table_bert(table_input_ids, attention_mask = table_input_mask, token_type_ids = table_token_type_ids).last_hidden_state
        table_outputs = torch.reshape(table_outputs, (-1, 768))


        segment_outputs = self.segment_relu(self.segment_linear(table_outputs))
        table_segment_embs = torch.matmul(segment_pooling_matrix, segment_outputs)

        graph_embs = torch.zeros(sent_embs.size()[0] + table_segment_embs.sizee()[0], 768).to(self.args.device)
        graph_embs[torch.nonzero(fusion_graph.ndata['t']==0).squeeze(1), :] = sent_embs
        if table_segment_embs.size() != torch.Size([0]):
            graph_embs[torch.nonzero(fusion_graph.ndata['t']==1).squeeze(1), :] = table_segment_embs
        assert fusion_graph.num_nodes() == fusion_graph_embs.size()[0], (fusion_graph.num_nodes(), fusion_graph_embs.size())

        fusion_graph_embs = self.fusion_gnn(fusion_graph, fusion_graph_embs)



        sent_h = fusion_graph_embs[:num_sentences]
        segment_h = fusion_graph_embs[num_sentences:]

        sent_hg = self.relu(self.dropout(self.sent_linear1(sent_h)))
        sent_pred_logits = torch.log_softmax(self.sent_linear2(sent_hg).view(-1, 2), dim=-1)

        segment_hg = self.relu(self.dropout(self.segment_linear1(segment_h)))
        segment_pred_logits = torch.log_softmax(self.segment_linear2(segment_hg).view(-1, 2), dim=-1)

        cell_outputs = self.cell_relu(self.cell_linear(table_outputs))

        # [batch_size, num_tables, num_cells , hidden_size]
        cell_graph_embs = torch.matmul(cell_pooling_matrix, cell_outputs)

        assert cell_graph.num_nodes() == cell_graph_embs.size()[0], print(cell_graph.num_nodes(),
                                                                          cell_graph_embs.size())

        cell_graph_embs = self.cell_gnn(cell_graph, cell_graph_embs)

        cell_hg = self.relu(self.dropout(self.cell_linear1(cell_graph_embs)))
        cell_pred_logits = torch.log_softmax(self.cell_linear2(cell_hg).view(-1, 2), dim=-1)

        return sent_pred_logits, segment_pred_logits, cell_pred_logits