import torch
from torch.utils.data import Dataset
import numpy as np
import inspect
import datasets
from transformers.data.metrics import glue_compute_metrics
from transformers import DataCollatorWithPadding, PreTrainedTokenizer

def load_attack_dataset(dataset_name: str):
    if dataset_name == 'sst':
        dataset = datasets.load_dataset("glue", "sst2")
        attack_set = dataset['validation']
        return attack_set
    elif dataset_name == 'mnli':
        dataset = datasets.load_dataset("glue", "mnli")
        attack_set = dataset['validation_matched']
        return attack_set
    elif dataset_name == 'qnli':
        dataset = datasets.load_dataset("glue", "qnli")
        attack_set = dataset['validation']
        attack_set = attack_set.rename_column("question", "premise")
        attack_set = attack_set.rename_column("sentence", "hypothesis")
        return attack_set
    elif dataset_name == 'rte':
        dataset = datasets.load_dataset("glue", "rte")
        attack_set = dataset['validation']
        attack_set = attack_set.rename_column("sentence1", "premise")
        attack_set = attack_set.rename_column("sentence2", "hypothesis")
        return attack_set
    elif dataset_name == 'agnews':
        dataset = datasets.load_dataset("ag_news")
        attack_set = dataset['test']
        attack_set = attack_set.rename_column("text", 'sentence')
        return dataset['test']
    else:
        raise NotImplementedError

def get_task_type(dataset_name):
    if dataset_name in ['sst', 'agnews']:
        return False
    elif dataset_name in ['rte','qnli','mnli']:
        return True

def get_class_num(dataset_name):
    if dataset_name in ['sst', 'rte', 'qnli']:
        return 2
    elif dataset_name in ['mnli']:
        return 3
    elif dataset_name in ['agnews']:
        return 4


def text_classification_metrics(task_name, preds, labels):
    return {"acc": (preds == labels).mean()}

compute_metrics_mapping = {
    "mnli": text_classification_metrics,
    "sst": text_classification_metrics,
    "agnews": text_classification_metrics,
    "qnli": text_classification_metrics,
    "rte": text_classification_metrics,
}

def remove_unused_columns(model, dataset: Dataset, reserved_columns = []):
    signature = inspect.signature(model.forward)
    _signature_columns = list(signature.parameters.keys())
    _signature_columns += ["label", "label_ids"]
    _signature_columns += reserved_columns
    columns = [k for k in _signature_columns if k in dataset.column_names]
    ignored_columns = list(set(dataset.column_names) - set(_signature_columns))
    return dataset.remove_columns(ignored_columns)


class LocalSSTDataset():
    def __init__(self, tokenizer = None) -> None:
        self.tokenizer = tokenizer
        dataset_dict = datasets.load_dataset("glue", "sst2")
        orig_train_set, valid_set, test_set = dataset_dict['train'],dataset_dict['validation'],dataset_dict['test']

        num_orig_train = len(orig_train_set['label'])
        num_new_train = int(num_orig_train * 0.9)
        rand_idxs = np.random.permutation(num_orig_train)
        rand_train_ids = rand_idxs[: num_new_train]
        rand_valid_ids = rand_idxs[num_new_train: ]

        test_set = valid_set
        train_set = orig_train_set.select(rand_train_ids)
        valid_set = orig_train_set.select(rand_valid_ids)


        self.train_dataset = train_set.map(self.tokenize_corpus, batched=True,)
        self.valid_dataset = valid_set.map(self.tokenize_corpus, batched=True,)
        self.test_dataset = test_set.map(self.tokenize_corpus, batched=True,)
        self.data_collator = DataCollatorWithPadding(tokenizer, padding = 'longest')


    def tokenize_corpus(self, examples):
        tokenized = self.tokenizer(examples['sentence'], truncation = True, max_length = 100)
        return tokenized

class LocalNLIDataset():
    def __init__(self, dataset_name = 'mnli', tokenizer = None) -> None:
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        if dataset_name == 'mnli':
            dataset_dict = datasets.load_dataset("glue", "mnli")
        elif dataset_name == 'qnli':
            dataset_dict = datasets.load_dataset("glue", "qnli")
            dataset_dict = dataset_dict.rename_column("question", "premise")
            dataset_dict = dataset_dict.rename_column("sentence", "hypothesis")
        elif dataset_name == 'rte':
            dataset_dict = datasets.load_dataset("glue", "rte")
            dataset_dict = dataset_dict.rename_column("sentence1", "premise")
            dataset_dict = dataset_dict.rename_column("sentence2", "hypothesis")
        else:
            raise NotImplementedError

        if dataset_name == 'mnli':
            orig_train_set, valid_set = dataset_dict['train'],dataset_dict['validation_matched']
        else:
            orig_train_set, valid_set = dataset_dict['train'],dataset_dict['validation']

        num_orig_train = len(orig_train_set['label'])
        num_new_train = int(num_orig_train * 0.9)
        rand_idxs = np.random.permutation(num_orig_train)
        rand_train_ids = rand_idxs[: num_new_train]
        rand_valid_ids = rand_idxs[num_new_train: ]

        test_set = valid_set
        train_set = orig_train_set.select(rand_train_ids)
        valid_set = orig_train_set.select(rand_valid_ids)

        self.train_dataset = train_set.map(self.tokenize_corpus, batched=True,)
        self.valid_dataset = valid_set.map(self.tokenize_corpus, batched=True,)
        self.test_dataset = test_set.map(self.tokenize_corpus, batched=True,)

        self.data_collator = DataCollatorWithPadding(tokenizer, padding = 'longest')

    def tokenize_corpus(self, examples):
        max_length = 100
        tokenized = self.tokenizer(examples['premise'], examples['hypothesis'], truncation = True, max_length = max_length, padding = 'longest')
        return tokenized


class LocalAGDataset():
    def __init__(self, tokenizer = None,) -> None:
        self.tokenizer = tokenizer
        dataset_dict = datasets.load_dataset("ag_news")
        dataset_dict = dataset_dict.rename_column("text", 'sentence')

        orig_train_set, test_set = dataset_dict['train'], dataset_dict['test']

        num_orig_train = len(orig_train_set['label'])
        num_new_train = int(num_orig_train * 0.9)
        rand_idxs = np.random.permutation(num_orig_train)
        rand_train_ids = rand_idxs[: num_new_train]
        rand_valid_ids = rand_idxs[num_new_train: ]

        test_set = test_set
        train_set = orig_train_set.select(rand_train_ids)
        valid_set = orig_train_set.select(rand_valid_ids)

        self.train_dataset = train_set.map(self.tokenize_corpus, batched=True,)
        self.valid_dataset = valid_set.map(self.tokenize_corpus, batched=True,)
        self.test_dataset = test_set.map(self.tokenize_corpus, batched=True,)
        self.data_collator = DataCollatorWithPadding(tokenizer, padding = 'longest')

    def tokenize_corpus(self, examples):
        tokenized = self.tokenizer(examples['sentence'], truncation = True, max_length = 100)
        return tokenized


