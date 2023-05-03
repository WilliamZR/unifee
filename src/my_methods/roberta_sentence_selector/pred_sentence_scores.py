import os
import torch
from tqdm import tqdm

from baseline.retriever.eval_sentence_retriever import eval_sentence_obj
from roberta_sentence_arg_parser import get_parser

from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from roberta_sentence_preprocessor import BasePreprocessor
from my_utils import set_seed, save_jsonl_data

from roberta_cls import RobertaCls
from roberta_sentence_generator import RobertaSentenceGenerator
import graph_cell_config as config

from database.feverous_db import FeverousDB



def main(args):
    if args is None:
        args = get_parser().parse_args()
    assert args.test_ckpt is not None
    set_seed(args.seed)

    print(args)

    tokenizer = AutoTokenizer.from_pretrained(args.bert_name)

    preprocessor = BasePreprocessor(args)

    args.label2id = config.label2idx
    args.id2label = dict(zip([value for _, value in args.label2id.items()], [key for key, _ in args.label2id.items()]))
    args.config = config
    args.tokenizer = tokenizer
    args.db = FeverousDB(args.wiki_path)

    if args.model_type == "RobertaCls":
        model = RobertaCls(args)
        data_generator = RobertaSentenceGenerator
    else:
        assert False, args.model_type

    args.test_mode = True

    # train_data, valid_data, test_data = preprocessor.process(
    #     args.data_dir, args.cache_dir, data_generator, args.tokenizer, dataset=["dev"])
    # train_data, valid_data, test_data = preprocessor.process(
    #     args.data_dir, args.cache_dir, data_generator, args.tokenizer, dataset=["dev", "train", "test"])
    train_data, valid_data, test_data = preprocessor.process(
        args.data_dir, args.cache_dir, data_generator, args.tokenizer, dataset=["test"])

    load_model_path = os.path.join(args.ckpt_root_dir, args.test_ckpt)

    ckpt_meta = model.load(load_model_path)
    model.to(args.device)

    if valid_data:
        dev_dataloader = DataLoader(valid_data, args.batch_size, shuffle=False)
        val(model, dev_dataloader, "dev", args)
    if train_data:
        train_dataloader = DataLoader(train_data, args.batch_size, shuffle=False)
        val(model, train_dataloader, "train", args)
    if test_data:
        test_dataloader = DataLoader(test_data, args.batch_size, shuffle=False)
        val(model, test_dataloader, "test", args)

@torch.no_grad()
def val(model, dataloader, data_type, args):
    """
    计算模型在验证集上的准确率等信息
    """
    dataloader.dataset.print_example()

    output_path = os.path.join(args.data_dir, f"{data_type}.sentences.roberta.p5.s5.jsonl" + "." + args.test_ckpt)
    print(f"save sentence scores to {output_path}")

    preds_epoch = []

    model.eval()
    for ii, data_entry in tqdm(enumerate(dataloader)):
        res = model(data_entry, args, test_mode=True)
        score_pos, _, golds = res
        preds = list(score_pos.cpu().detach().numpy())
        preds_epoch.extend(preds)

    cand_id_lst = dataloader.dataset.cand_id_lst
    assert len(preds_epoch) == sum([len(ci) for ci in cand_id_lst]), print(len(preds_epoch), sum([len(ci) for ci in cand_id_lst]))
    res = []
    sent_scores = []
    stat = 0
    preds_epoch = [float(p) for p in preds_epoch]
    for cand_ids in cand_id_lst:
        pred_scores = preds_epoch[stat:stat+len(cand_ids)]
        preds = list(torch.topk(torch.tensor(pred_scores), k=min(5, len(pred_scores)))[1].numpy())
        pred_ids = [cand_ids[idx] for idx in preds]
        sent_scores.append(list(zip(cand_ids, pred_scores)))
        res.append(pred_ids)
        stat += len(cand_ids)

    odata = []
    for entry, preds, sent_score in zip(dataloader.dataset.raw_data, res, sent_scores):
        oentry = {
            "id": entry["id"],
            "claim": entry["claim"],
            "predicted_sentences": preds,
            "sentence_scores": sent_score
        }
        odata.append(oentry)
    if not data_type == "test":
        eval_sentence_obj(odata, data_type)
    save_jsonl_data(odata, output_path)
    return res

if __name__ == '__main__':
    main(None)