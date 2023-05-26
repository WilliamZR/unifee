import torch
from tqdm import tqdm

from fusion_col_parser import get_parser
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from base_templates import BasePreprocessor
from my_utils.pytorch_common_utils import set_seed, get_optimizer 
from my_utils.task_metric import compute_metrics
from my_utils.common_utils import average, save_pkl_data
from my_utils.torch_model_utils import print_grad
from transformers import get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from fusion_col_util import build_dataset, collate_fn, FusionRetrievalDataset
from fusion_evidence_eval_matrix import EvalMetric
from fusion_col_models import FusionEvidenceModel
from database.feverous_db import FeverousDB
from all_cell_util import build_dataset_with_all_cells

def compute_cell_probabity(seg_logits, cell_logits, table_ids_2D, cell_ids):
    output = []
    seg_logits = np.exp(seg_logits)
    cell_logits = np.exp(cell_logits)
    seg_step = 0
    cell_step = 0
    for i in range(len(cell_ids)):
        col_num = len(table_ids_2D[i][0])
        temp_seg_logits = seg_logits[seg_step : seg_step + col_num]
        temp_cell_logit = cell_logits[cell_step : cell_step + len(cell_ids[i])]

        for j, id in enumerate(cell_ids[i]):
            col_id = int(id[-1])
            output.append((id, temp_seg_logits[col_id, 1] * temp_cell_logit[j, 1]))

        seg_step += col_num
        cell_step += len(cell_ids[i])
    return output

def main():
    args = get_parser().parse_args()
    print(args)
    args.db = FeverousDB(args.wiki_path)
    args.output_path = 'data/dev.allcell.scores.jsonl'

    valid_data = build_dataset_with_all_cells('dev', args, cache_file = 'data/dev_all_cells.jsonl')
    val_dataset = FusionRetrievalDataset(valid_data, args)
    print(len(val_dataset))
    val_dataloader = DataLoader(val_dataset, batch_size = 1, collate_fn=collate_fn, shuffle=False)

    model = torch.load(args.model_load_path)
    model.to(args.device)
    model.eval()


    scores = {}
    with torch.no_grad():
        for ii, batch in tqdm(enumerate(val_dataloader)):
            batch = [item.to(args.device) for item in batch]
            input_data = batch[:-3]
            res = model(input_data, args)

            sent_pred_logits, segment_pred_logits, cell_pred_logits = res

            sent_pred_logits = sent_pred_logits.detach().cpu().numpy()
            segment_pred_logits = segment_pred_logits.detach().cpu().numpy()
            cell_pred_logits = cell_pred_logits.detach().cpu().numpy()

            sent_ids = valid_data['sent_ids'][ii]
            cell_ids = valid_data['cell_ids'][ii]
            table_ids_2D = valid_data['table_ids_2D'][ii]
            evidence_id = valid_data['id'][ii]

            cell_scores = compute_cell_probabity(segment_pred_logits, cell_pred_logits, table_ids_2D, cell_ids)
            sent_scores = list(zip(sent_ids, sent_pred_logits))
            scores[evidence_id] = (cell_scores, sent_scores)
    save_pkl_data(scores, args.output_path)
if __name__ == '__main__':
    main()