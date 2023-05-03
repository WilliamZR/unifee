import os
import torch
from tqdm import tqdm

from baseline.retriever.eval_sentence_retriever import eval_sentence_obj
from roberta_sentence_arg_parser import get_parser

from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch.nn as nn

from roberta_sentence_preprocessor import BasePreprocessor
from my_utils import set_seed, get_optimizer, compute_metrics, save_jsonl_data
from my_utils import print_grad, average
from transformers import get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from my_utils import generate_data4debug

from roberta_cls import RobertaCls
from roberta_sentence_generator import RobertaSentenceGenerator
import graph_cell_config as config

from database.feverous_db import FeverousDB

os.chdir("/home/hunan/feverous/mycode")

def main():
    args = get_parser().parse_args()
    print(args)
    set_seed(args.seed)

    generate_data4debug(args.data_dir)
    if "small_" in args.data_dir:
        # args.force_generate = True
        args.batch_size = 2
        args.gradient_accumulation_steps = 2

    tokenizer = AutoTokenizer.from_pretrained(args.bert_name)

    preprocessor = BasePreprocessor(args)

    args.label2id = config.label2idx
    args.id2label = dict(zip([value for _, value in args.label2id.items()], [key for key, _ in args.label2id.items()]))
    args.config = config
    args.tokenizer = tokenizer
    args.db = FeverousDB(args.wiki_path)
    args.test_mode = False

    if args.model_type == "RobertaCls":
        model = RobertaCls(args)
        data_generator = RobertaSentenceGenerator
    else:
        assert False, args.model_type

    train_data, valid_data, test_data = preprocessor.process(
        args.data_dir, args.cache_dir, data_generator, args.tokenizer, dataset=["train", "dev"])

    train_dataloader = DataLoader(train_data, args.batch_size, shuffle=True)
    val_dataloader = DataLoader(valid_data, args.batch_size, shuffle=False)
    if test_data:
        test_dataloader = DataLoader(test_data, args.batch_size, shuffle=False)

    if args.load_model_path:
        model.load(args.load_model_path)

    model.to(args.device)
    tb = SummaryWriter()
    optimizer = get_optimizer(model, lr=args.lr, weight_decay=args.weight_decay, fix_bert=False)

    total_steps = int(args.max_epoch * len(train_dataloader)) // args.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(optimizer
                                                , num_warmup_steps=int(total_steps * args.warm_rate)
                                                , num_training_steps=total_steps)

    criterion = nn.MarginRankingLoss(margin=1)

    global_step = 0
    optimizer.zero_grad()

    for epoch in range(args.max_epoch):
        train_data.generate_train_instances_one_epoch()
        model.train()
        acc_sum = 0
        loss_sum = 0
        prec_sum = 0
        recall_sum = 0

        preds_epoch = []
        golds_epoch = []
        loss_epoch = []
        train_dataloader.dataset.print_example()
        for ii, batch in tqdm(enumerate(train_dataloader)):
            # train model
            try:
                res = model(batch, args, test_mode=False)
            except RuntimeError as e:
                torch.cuda.empty_cache()
                res = model(batch, args, test_mode=False)

            score_pos, score_neg, golds = res
            loss = criterion(score_pos, score_neg, golds)

            # preds = logits.topk(k=1, dim=-1)[1].squeeze(-1)
            # preds_epoch.extend(list(preds.cpu().detach().numpy()))

            pred_scores = list(torch.cat([score_pos, score_neg], dim=0).cpu().detach().numpy())
            preds = [1 if item > args.predict_threshold else 0 for item in pred_scores]
            golds = [1] * len(batch[0]) + [0]*len(batch[0])
            preds_epoch.extend(preds)
            golds_epoch.extend(golds)
            loss_epoch.append(loss.item())

            acc_sum += sum([1 if p == g else 0 for p, g in zip(preds, golds)]) / len(preds)
            recall_sum += len([1 for p, g in zip(preds, golds) if p == g and p == 1]) / sum(golds)
            prec_sum += len([1 for p, g in zip(preds, golds) if p == g and p == 1]) / sum(preds) if sum(preds) else 1

            global_step += 1
            loss = loss / args.gradient_accumulation_steps
            loss_sum += loss.item()
            loss.backward()

            if global_step % args.gradient_accumulation_steps == 0:
                lrs = scheduler.get_last_lr()
                tb.add_scalars("learning_rates", {"bert_lr": lrs[0], "no_bert_lr": lrs[-1]}, global_step)
                tb.add_scalar("train_loss", loss.item(), global_step)
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=5.0)
                optimizer.step()
                scheduler.step()

                grad_dict_first = print_grad(model)
                tb.add_scalars("model_grads_first", grad_dict_first, global_step)
                '''
                grad_dict_second = print_grad(model, "second")
                tb.add_scalars("model_grads_second", grad_dict_second, global_step)
                '''
                optimizer.zero_grad()

            if ii % args.print_freq == 0:
                print('Epoch:{0},step:{1}'.format(epoch, ii))
                freq = args.print_freq
                print('Train Loss:{:.6f}    Acccuracy:{:.3f}   Precision:{:3f}   Recall {:3f}'.format(
                    loss_sum/freq, acc_sum*100/freq, prec_sum*100/freq, recall_sum*100/freq))
                loss_sum = 0
                acc_sum = 0
                prec_sum = 0
                recall_sum = 0

        print("====train step of epoch {} ==========".format(epoch))
        loss, acc, prec, recall = print_res(loss_epoch, preds_epoch, golds_epoch, "train", epoch)

        # validate
        print("====validation step of epoch {}======================".format(epoch))
        val_preds, val_recall = val(model, val_dataloader, criterion, tokenizer, preprocessor, tb, epoch, args)

        ckpt_meta = {
            "train_loss": loss,
            "train_acc": acc,
            "train_prec" : prec,
            "train_recall": recall,
            # "val_loss": val_loss,
            # "val_acc": val_acc,
            # "val_prec": val_prec,
            # "val_recall": val_recall,
            "data_generator": args.data_generator,
        }

        path = model.save(args.ckpt_root_dir, ckpt_meta, val_recall, only_max=(not args.save_all_ckpt))
        if path:
            tokenizer.save_vocabulary(path)
            try:
                model.config.save_pretrained(path)
            except:
                pass
            if args.test:
                # TODO: save test results
                print("==============test step of epoch {}=================".format(epoch))
                # test_loss, test_acc = val(model, test_dataloader, criterion, tokenizer, preprocessor, tb, epoch, args)
    tb.flush()
    tb.close()
    args.test_ckpt = model.subdir
    print("the testing checkpoint is", args.test_ckpt)
    import pred_sentence_scores
    pred_sentence_scores.main(args)

@torch.no_grad()
def val(model, dataloader, criterion, tokenizer, preprocessor, tb, epoch, args):
    """
    计算模型在验证集上的准确率等信息
    """
    dataloader.dataset.print_example()
    preds_epoch = []

    model.eval()
    for ii, data_entry in tqdm(enumerate(dataloader)):
        res = model(data_entry, args, test_mode=True)
        score_pos, _, golds = res
        preds = score_pos.cpu().detach().numpy()
        # preds = [1 if item > args.predict_threshold else 0 for item in list(score_pos.cpu().detach().numpy())]

        preds_epoch.extend(preds)

    cand_id_lst = dataloader.dataset.cand_id_lst
    assert len(preds_epoch) == sum([len(ci) for ci in cand_id_lst]), print(len(preds_epoch), sum([len(ci) for ci in cand_id_lst]))
    res = []
    sent_scores = []
    stat = 0
    for cand_ids in cand_id_lst:
        pred_scores = preds_epoch[stat:stat+len(cand_ids)]
        preds = list(torch.topk(torch.tensor(pred_scores), k=min(10, len(pred_scores)))[1].numpy())
        pred_ids = [cand_ids[idx] for idx in preds]
        sent_scores.append(list(zip(cand_ids, pred_scores)))
        res.append(pred_ids)
        stat += len(cand_ids)

    odata = []
    for entry, preds in zip(dataloader.dataset.raw_data, res):
        oentry = {
            "id": entry["id"],
            "claim": entry["claim"],
            "predicted_sentences": preds,
            "sentence_scores": sent_scores
        }
        odata.append(oentry)
    val_recall = eval_sentence_obj(odata, "dev")
    # save_jsonl_data(odata, "/home/hunan/feverous/mycode/data/dev.sentences.roberta.p5.s5.jsonl")
    return res, val_recall

def print_res(loss_epoch, preds, golds, data_type, epoch):
    loss = average(loss_epoch)
    acc = sum([1 if p == g else 0 for p, g in zip(preds, golds)]) / len(preds)
    recall = len([1 for p, g in zip(preds, golds) if p == g and p == 1]) / sum(golds)
    prec = len([1 for p, g in zip(preds, golds) if p == g and p == 1]) / sum(preds) if sum(preds) else 1
    print("{}_epoch{}_loss:".format(data_type, epoch), loss)
    print("{}_epoch{}_accuracy:".format(data_type, epoch), acc)
    print("{}_epoch{}_precision:".format(data_type, epoch), prec)
    print("{}_epoch{}_recall:".format(data_type, epoch), recall)
    scores = compute_metrics(preds, golds)
    return loss, acc, prec, recall


if __name__ == "__main__":
    main()
