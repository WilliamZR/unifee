Code for [Unified Evidence Extraction for Fact Verification over Tables and Texts](https://aclanthology.org/2023.eacl-main.82/) at EACL2023.

## Shared Task
The majority of the [base-line code and its documentation](https://github.com/Raldir/FEVEROUS) remains intact. We extend it by adding scripts to incorporate neural re-ranking models.

## Install Requirements 
Create a new Conda environment and install torch:
```
conda env freate -f feverous.yaml
```

## Download PLM checkpoints
Download RoBERTa TAPAS from huggingface

## Prepare Data
Call the following script to download the FEVEROUS data:
```
./scripts/download_data.sh 
```
Or you can download the data from the [FEVEROUS dataset page](https://fever.ai/dataset/feverous.html) directly. Namely:

* Training Data
* Development Data
* Test Data
* Wikipedia Data as a database (sqlite3)

unpack the given downloaded data to DCUF_code/data/, and rename them to
* train.jsonl
* dev.jsonl
* test.jsonl.bk
* feverous_wikiv1.db

## Running the Code

### prepare the test data
add an id to each test case to make it have the same format as other splits
```
cd src/my_script/
python add_id_to_test_set.py
```

### Page Retriever
See src/my_methods/bm25_doc_retriever/readme.md for the Page Retriever step

### Sentence and Table Evidence Retrieval
The top l sentences and q tables of the selected pages are then scored separately using TF-IDF. We set l=5 and q=3.

Extract sentence evidence
```
PYTHONPATH=src python  src/my_methods/roberta_sentence_selector/train_roberta_sentence_selector.py --bert_name {bert_name} > log_graph_sentence_selector_large.txt &
PYTHONPATH=src python  src/my_methods/roberta_sentence_selector/pred_sentence_scores.py --test_ckpt {test_ckpt} 
```

Extract table evidence
```
PYTHONPATH=src python src/baseline/retriever/sentence_tfidf_drqa.py --db data/feverous_wikiv1.db --max_page 5 --max_sent 5 --use_precomputed false --data_path data/ --split {split}
PYTHONPATH=src python src/baseline/retriever/table_tfidf_drqa.py --db data/feverous_wikiv1.db --max_page 5 --max_tabs 3 --use_precomputed false --data_path data/ --split {split}
```

check the results of table evidence
```
PYTHONPATH=src python src/baseline/retriever/eval_tab_retriever.py --max_page 5 --max_tabs 3 --split {split} --input_path {input_path}
```


Check the results of sentence evidence
```
PYTHONPATH=src python src/baseline/retriever/eval_sentence_retriever.py --max_page 5 --max_sent 5 --split {split}
```

Combine both retrieved sentences and tables into one file:
 ```
 PYTHONPATH=src python src/baseline/retriever/combine_retrieval.py --data_path data --max_page 5 --max_sent 5 --max_tabs 3 --split {split}
 ```

Evaluate combined results:
```
PYTHONPATH=src python src/baseline/retriever/eval_combined_retriever.py --max_page 5 --max_sent 5 --max_tabs 3 --split {split}
```

Build dataset, prepare graphs for each split
```
PYTHONPATH=src python src/my_methods/graph_evidence_extraction/all_cell_util.py --split {split}
```

Train
```
train_fusion_col_extractor.py --lr 1e-6 --batch_size 4 --print_freq 100   --use_entity_edges --max_epoch 3 --use_all_cells 
```

Run model on dataset and save scores to output_path
```
rerank_evidence.py  --output_path {} --model_load_path {} --use_entity_edges
```

Retrieve evidence set with threshold from computed score
```
python evidence_retrieval_with_scores.py --split {} --cell_threshold {} --sent_threshold {}
```

Verdict Prediction please refer to our previous work [DCUF](https://github.com/lanlanabcd/dual_channel_feverous)