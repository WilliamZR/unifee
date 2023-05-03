import argparse
import os
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default = 'data',help='/path/to/data')
    parser.add_argument('--roberta_path', type = str)
    parser.add_argument('--tapas_path', type = str)
    parser.add_argument('--model_save_path', type = str)
    parser.add_argument('--output_path', type = str)
    parser.add_argument('--device',type=str,default='cuda:1')
    parser.add_argument('--model_load_path', type =str)
    parser.add_argument('--split', type = str)

    parser.add_argument('--wiki_path',default='data/feverous_wikiv1.db',type=str)

    parser.add_argument('--max_epoch',type=int, default = 6)

    parser.add_argument('--warm_rate', type=int,default = 0)
    parser.add_argument("--lr", default=1e-7, type=float)

    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float, help='Weight decay (L2 loss on parameters).')
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--print_freq",type = int,default=100)
    
    parser.add_argument('--sent_threshold', type = float, default= 0.15)
    parser.add_argument('--seg_threshold', type = float, default = 0.2)
    parser.add_argument('--cell_threshold', type = float, default = 0.15)
    
    parser.add_argument('--dropout', type = float, default = 0.2)

    parser.add_argument('--seed', default = 1234, type = int)

    parser.add_argument('--small_dataset', action = 'store_true')
    
    parser.add_argument('--use_entity_edges', action = 'store_true')
    parser.add_argument('--use_all_cells', action = 'store_true')
    parser.add_argument('--fix_bert', action = 'store_true')

    return parser