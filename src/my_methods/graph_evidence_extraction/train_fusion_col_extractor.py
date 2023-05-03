import os
import torch
from tqdm import tqdm

from fusion_col_parser import get_parser
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch.nn as nn

from base_templates import BasePreprocessor
from my_utils.pytorch_common_utils import set_seed, get_optimizer 
from my_utils.task_metric import compute_metrics
from my_utils.common_utils import average
from my_utils.torch_model_utils import print_grad
from transformers import get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from fusion_col_util import build_dataset, collate_fn, FusionRetrievalDataset
from fusion_evidence_eval_matrix import EvalMetric
from fusion_col_models import FusionEvidenceModel
from database.feverous_db import FeverousDB
from all_cell_util import build_dataset

def main():
    use_schedular = True
    args = get_parser().parse_args()

    print(args)
    set_seed(args.seed)

    args.label2id = 2
    args.db = FeverousDB(args.wiki_path)

    if args.use_all_cells:
        print('Use All The Cells as Candiates')
        train_data = build_dataset('train', args, 'data/train_all_cells.jsonl')

        valid_data = build_dataset('dev', args, 'data/dev_all_cells.jsonl')


    train_dataset = FusionRetrievalDataset(train_data, args)
    val_dataset = FusionRetrievalDataset(valid_data, args)

    train_dataloader = DataLoader(train_dataset, args.batch_size, collate_fn=collate_fn, shuffle=True)
    val_dataloader = DataLoader(val_dataset, args.batch_size, collate_fn=collate_fn, shuffle=False)


    model = FusionEvidenceModel(args).to(args.device)
 
    tb = SummaryWriter()
    optimizer = get_optimizer(model, lr=args.lr, weight_decay=args.weight_decay, fix_bert= args.fix_bert)

    if use_schedular:
        total_steps = int(args.max_epoch * len(train_dataloader)) // args.gradient_accumulation_steps
        scheduler = get_linear_schedule_with_warmup(optimizer
                                                    , num_warmup_steps=int(total_steps * args.warm_rate)
                                                    , num_training_steps=total_steps)

    cell_criterion = nn.NLLLoss(weight=torch.tensor([0.1,0.4]).to(args.device))
    sent_criterion = nn.NLLLoss()
    segment_criterion = nn.NLLLoss()

    global_step = 0
    optimizer.zero_grad()

    sent_metric = EvalMetric()
    segment_metric = EvalMetric()
    cell_metric = EvalMetric()
    for epoch in range(args.max_epoch):
        model.train()
        sent_pred_epoch = []
        sent_gold_epoch = []
        segment_pred_epoch = []
        segment_gold_epoch = []
        cell_pred_epoch = []
        cell_gold_epoch = []
        loss_sum = 0

        loss_epoch = []
        for ii, batch in tqdm(enumerate(train_dataloader)):
            # train model
            batch = [item.to(args.device) for item in batch]
            input_data, labels = batch[:-3], batch[-3:]
            try:
                res = model(input_data, args)
            except RuntimeError as e:
                torch.cuda.empty_cache()
                res = model(input_data, args)

            # logits [batch_size, num_classes]
            sent_pred_logits, segment_pred_logits, cell_pred_logits = res
            sent_labels, segment_labels, cell_labels = labels

            sent_loss = sent_criterion(sent_pred_logits, sent_labels)
            segment_loss = segment_criterion(segment_pred_logits, segment_labels)
            cell_loss = cell_criterion(cell_pred_logits, cell_labels)
            loss = sent_loss + segment_loss + cell_loss

            acc, recall, precision = cal_matrix(sent_pred_logits, sent_labels, sent_pred_epoch, sent_gold_epoch, args.sent_threshold, args)
            sent_metric.meter_add(acc, recall, precision, sent_loss.item())

            acc, recall, precision = cal_matrix(segment_pred_logits, segment_labels, segment_pred_epoch, segment_gold_epoch, args.seg_threshold, args)
            segment_metric.meter_add(acc, recall, precision, segment_loss.item())

            acc, recall, precision = cal_matrix(cell_pred_logits, cell_labels, cell_pred_epoch, cell_gold_epoch, args.cell_threshold, args)
            cell_metric.meter_add(acc, recall, precision, cell_loss.item())

            loss_epoch.append(loss.item())

            global_step += 1
            loss = loss / args.gradient_accumulation_steps
            loss_sum += loss.item()
            loss.backward()

            if global_step % args.gradient_accumulation_steps == 0:
                if use_schedular:
                    lrs = scheduler.get_last_lr()
                    tb.add_scalars("learning_rates", {"bert_lr": lrs[0], "no_bert_lr": lrs[-1]}, global_step)
                tb.add_scalar("train_loss", loss.item(), global_step)
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=5.0)
                optimizer.step()
                if use_schedular:
                    scheduler.step()

                grad_dict_first = print_grad(model)
                tb.add_scalars("model_grads_first", grad_dict_first, global_step)

                optimizer.zero_grad()

            if ii % args.print_freq == 0:
                print('Epoch:{0},step:{1}'.format(epoch, ii))
                freq = args.print_freq
                print('Train Loss:{:.6f}'.format(loss_sum/freq))
                print('Sentences')
                sent_metric.print_meter()
                print('Columns')
                segment_metric.print_meter()
                print('Cells')
                cell_metric.print_meter()

                loss_sum = 0
                sent_metric.meter_reset()
                segment_metric.meter_reset()
                cell_metric.meter_reset()

        print("====train step of epoch {} ==========".format(epoch))
        loss, acc, prec, recall = print_res(loss_epoch
                                            , sent_pred_epoch, sent_gold_epoch
                                            , segment_pred_epoch, segment_gold_epoch
                                            , cell_pred_epoch, cell_gold_epoch
                                            , "train", epoch)

        # validate
        print("====validation step of epoch {}======================".format(epoch))
       
        val(model, val_dataloader, [sent_criterion, segment_criterion, cell_criterion], tb, epoch, args)

        torch.save(model, args.model_save_path)



def cal_matrix(pred_logits, golds, pred_epoch, gold_epoch, predict_threshold, args):
    assert len(pred_logits) == len(golds)
    golds = list(golds.cpu().detach().numpy())
    golds = [g for g in golds if g != -100]

    pred_scores = list(torch.exp(pred_logits[:len(golds)])[:, 1].cpu().detach().numpy())
    preds = [1 if item > predict_threshold else 0 for item in pred_scores]

    pred_epoch.extend(preds)
    gold_epoch.extend(golds)

    acc = sum([1 if p == g else 0 for p, g in zip(preds, golds)]) / len(preds)
    recall = len([1 for p, g in zip(preds, golds) if p == g and p == 1]) / sum(golds) if sum(golds) else 1
    precision = len([1 for p, g in zip(preds, golds) if p == g and p == 1]) / sum(preds) if sum(preds) else 1
    return acc, recall, precision

@torch.no_grad()
def val(model, dataloader, criterion, tb, epoch, args):
    loss_sum = 0
    sent_pred_epoch = []
    sent_gold_epoch = []
    segment_pred_epoch = []
    segment_gold_epoch = []
    cell_pred_epoch = []
    cell_gold_epoch = []
    val_sent_metric = EvalMetric()
    val_seg_metric = EvalMetric()
    val_cell_metric = EvalMetric()

    loss_epoch = []

    sent_criterion, segment_criterion, cell_criterion = criterion

    model.eval()
    for ii, batch in tqdm(enumerate(dataloader)):
        batch = [item.to(args.device) for item in batch]
        input_data, labels = batch[:-3], batch[-3:]
        res = model(input_data, args)

        # logits [batch_size, num_classes]
        sent_pred_logits, segment_pred_logits, cell_pred_logits = res
        sent_labels, segment_labels, cell_labels = labels
        
        sent_loss = sent_criterion(sent_pred_logits, sent_labels)
        segment_loss = segment_criterion(segment_pred_logits, segment_labels)
        cell_loss = cell_criterion(cell_pred_logits, cell_labels)
        loss = sent_loss + segment_loss + cell_loss
        loss_epoch.append(loss.item())

        acc, recall, precision = cal_matrix(sent_pred_logits, sent_labels, sent_pred_epoch, sent_gold_epoch, args.sent_threshold, args)
        val_sent_metric.meter_add(acc, recall, precision, sent_loss.item())
        acc, recall, precision = cal_matrix(segment_pred_logits, segment_labels, segment_pred_epoch, segment_gold_epoch, args.seg_threshold, args)
        val_seg_metric.meter_add(acc, recall, precision, segment_loss.item())

        acc, recall, precision = cal_matrix(cell_pred_logits, cell_labels, cell_pred_epoch, cell_gold_epoch, args.cell_threshold, args)
        val_cell_metric.meter_add(acc, recall, precision, cell_loss.item())

    val_sent_metric.print_meter()
    val_seg_metric.print_meter()
    val_cell_metric.print_meter()
    #return loss_epoch, sent_pred_epoch, sent_gold_epoch, segment_pred_epoch, segment_gold_epoch, cell_pred_epoch, cell_gold_epoch

def print_res(loss_epoch, sent_preds, sent_golds, seg_preds, seg_golds, cell_preds, cell_golds, data_type, epoch):
    loss = average(loss_epoch)
    print("{}_epoch{}_loss:".format(data_type, epoch), loss)

    print("sentence metrics:")
    preds = sent_preds
    golds = sent_golds
    acc = sum([1 if p == g else 0 for p, g in zip(preds, golds)]) / len(preds)
    recall = len([1 for p, g in zip(preds, golds) if p == g and p == 1]) / sum(golds)
    prec = len([1 for p, g in zip(preds, golds) if p == g and p == 1]) / sum(preds) if sum(preds) else 1
    print("{}_epoch{}_accuracy:".format(data_type, epoch), acc)
    print("{}_epoch{}_precision:".format(data_type, epoch), prec)
    print("{}_epoch{}_recall:".format(data_type, epoch), recall)
    scores = compute_metrics(preds, golds)

    print("segment metrics:")
    preds = seg_preds
    golds = seg_golds
    acc = sum([1 if p == g else 0 for p, g in zip(preds, golds)]) / len(preds)
    recall = len([1 for p, g in zip(preds, golds) if p == g and p == 1]) / sum(golds)
    prec = len([1 for p, g in zip(preds, golds) if p == g and p == 1]) / sum(preds) if sum(preds) else 1
    print("{}_epoch{}_accuracy:".format(data_type, epoch), acc)
    print("{}_epoch{}_precision:".format(data_type, epoch), prec)
    print("{}_epoch{}_recall:".format(data_type, epoch), recall)
    scores = compute_metrics(preds, golds)

    print("cell metrics:")
    preds = cell_preds
    golds = cell_golds
    acc = sum([1 if p == g else 0 for p, g in zip(preds, golds)]) / len(preds)
    recall = len([1 for p, g in zip(preds, golds) if p == g and p == 1]) / sum(golds)
    prec = len([1 for p, g in zip(preds, golds) if p == g and p == 1]) / sum(preds) if sum(preds) else 1
    print("{}_epoch{}_accuracy:".format(data_type, epoch), acc)
    print("{}_epoch{}_precision:".format(data_type, epoch), prec)
    print("{}_epoch{}_recall:".format(data_type, epoch), recall)
    scores = compute_metrics(preds, golds)

    return loss, acc, prec, recall


if __name__ == "__main__":
    main()
