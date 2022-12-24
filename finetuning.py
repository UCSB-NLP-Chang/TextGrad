import dataclasses
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional
import logging
import time
import os
import numpy as np
import torch

import transformers
transformers.logging.set_verbosity_error()
from transformers import (
    RobertaTokenizer, RobertaForSequenceClassification, AdamW, 
    BertTokenizer, BertForSequenceClassification,
    AlbertTokenizer, AlbertForSequenceClassification,
    EvalPrediction,
    )
from transformers import HfArgumentParser, TrainingArguments, set_seed, Trainer

from src.data_util.dataloader import (LocalSSTDataset, LocalAGDataset, LocalNLIDataset, 
                                        remove_unused_columns, compute_metrics_mapping,
                                        get_class_num
                                        )

logger = logging.getLogger(__name__)
MODEL_CACHE_DIR = './model_cache/'

@dataclass
class MyArguments:
    # use_wandb: bool = field(default = False)
    dataset: str = field(default = 'sst')    ## choices: sst/mnli/qnli/rte/agnews
    model: str = field(default = 'roberta')  ## choices: bert/roberta/albert

if __name__ == '__main__':
    parser = HfArgumentParser((MyArguments, TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()
    dataset = args.dataset
    model_type = args.model

    num_labels = get_class_num(dataset)

    if training_args.report_to == ['wandb']:
        use_wandb = True
    else:
        use_wandb = False
    if use_wandb:
        wandb_name = f"finetuning-{model_type}-{dataset}"
        os.environ["WANDB_PROJECT"] = f'textgrad-finetuning'
        training_args.run_name = wandb_name
        training_args.report_to = 'wandb',   ## parameters:  run_name;  to set project name, use os.environ["WANDB_PROJECT"] = "huggingface"
    # else:
        # os.environ["WANDB_DISABLED"] = "true"
        # training_args.report_to = []

    if model_type == 'roberta':
        cache_dir = MODEL_CACHE_DIR + 'roberta_model/roberta-large/'
        tokenizer = RobertaTokenizer.from_pretrained('roberta-large', cache_dir = cache_dir)
        model = RobertaForSequenceClassification.from_pretrained('roberta-large', cache_dir = cache_dir, num_labels = num_labels)
        model_fn = RobertaForSequenceClassification
    elif model_type == 'bert':
        cache_dir = MODEL_CACHE_DIR + 'bert_model/bert-base-uncased/'
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', cache_dir = cache_dir, num_labels = num_labels)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir = cache_dir)
        model_fn = BertForSequenceClassification
    elif model_type == 'albert':
        cache_dir = MODEL_CACHE_DIR + 'albert_model/albert-xxlarge-v2/'
        tokenizer = AlbertTokenizer.from_pretrained('albert-xxlarge-v2', cache_dir = cache_dir)
        model = AlbertForSequenceClassification.from_pretrained('albert-xxlarge-v2', cache_dir = cache_dir, num_labels = num_labels)
        model_fn = AlbertForSequenceClassification
    else:
        raise NotImplementedError

    if dataset in ['rte', 'qnli', 'mnli']:
        all_dataset = LocalNLIDataset(dataset_name = dataset, tokenizer = tokenizer)
    elif dataset == 'agnews':
        all_dataset = LocalAGDataset(tokenizer = tokenizer)
    elif dataset == 'sst':
        all_dataset = LocalSSTDataset(tokenizer = tokenizer)
    else:
        raise NotImplementedError
    train_dataset = all_dataset.train_dataset
    valid_dataset = all_dataset.valid_dataset
    test_dataset = all_dataset.test_dataset

    train_dataset = remove_unused_columns(model, train_dataset)
    valid_dataset = remove_unused_columns(model, valid_dataset)
    test_dataset = remove_unused_columns(model, test_dataset)

    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            # Note: the eval dataloader is sequential, so the examples are in order.
            # We average the logits over each sample for using demonstrations.
            predictions = p.predictions
            logits = predictions            
            preds = np.argmax(logits, axis=1)
            label_ids = p.label_ids
            return compute_metrics_mapping[task_name](task_name, preds, label_ids)

        return compute_metrics_fn
    
    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = valid_dataset,
        compute_metrics = build_compute_metrics_fn(args.dataset),
        data_collator = all_dataset.data_collator,
    )

    if training_args.do_train:
        trainer.train()
        # Reload the best checkpoint (for eval)
        # model = model_fn.from_pretrained(training_args.output_dir)
        # model = model.to(training_args.device)
        # trainer.model = model
        # model.tokenizer = tokenizer

    # Evaluation
    final_result = {
    }

    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Validate ***")

        eval_datasets = [test_dataset]

        for eval_dataset in eval_datasets:
            output = trainer.evaluate(eval_dataset=eval_dataset)
            # eval_result = output.metrics 
            eval_results.update(output)

    print(eval_results)


