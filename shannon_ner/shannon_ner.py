# encoding: utf-8
"""
Model Description

@author: Xiaofei Sun
@contact: adoni1203@gmail.com
@version: 0.1
@license: Apache Licence
@file: interactive_example.py
@time: 2019-02-11
"""

import argparse
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange

from pytorch_pretrained_bert.modeling import BertForTokenClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple token classification."""

    def __init__(self, guid=None, text=None, labels=None, predicts=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text: list. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            labels: (Optional) list. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.labels = labels
        self.predicts = predicts


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    def __init__(self, data_dir, do_lower_case):
        self.all_labels = []
        self.data_dir = data_dir
        self.do_lower_case = do_lower_case

    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        return self._create_examples(os.path.join(self.data_dir, 'train.char.bmes'), 'train')

    def get_dev_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        return self._create_examples(os.path.join(self.data_dir, 'test.char.bmes'), 'dev')

    def _create_examples(self, file_name, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        last_tokens = []
        last_labels = []
        with open(file_name) as fi:
            for (i, line) in enumerate(fi):
                if line.strip():
                    token, label = line.strip().split()
                    if self.do_lower_case:
                        token = token.lower()
                    last_tokens.append(token)
                    last_labels.append(label)
                    if label not in self.all_labels:
                        self.all_labels.append(label)
                else:
                    examples.append(InputExample(guid="%s-%s" % (set_type, i), text=last_tokens, labels=last_labels))
                    last_tokens = []
                    last_labels = []
        return examples


def wrap_a_sentence_to_examples(text):
    return [InputExample(guid="0", text=list(text))]


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens = example.text[:(max_seq_length - 2)] if len(example.text) > max_seq_length - 2 else example.text
        if example.labels:
            labels = example.labels[:(max_seq_length - 2)] if len(example.labels) > max_seq_length - 2 else example.labels
        else:
            labels = []
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        tokens = [t if t in tokenizer.vocab else "[UNK]" for t in tokens]
        if example.labels:
            labels = label_list[:1] + labels + label_list[:1]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        label_ids = [label_list.index(i) for i in labels]
        segment_ids = [0] * len(tokens)
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        label_ids += padding
        input_mask += padding
        segment_ids += padding
        assert len(segment_ids) == len(input_mask) == len(input_ids) == max_seq_length

        features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_ids=label_ids))
    return features


def reverse_style(input_string):
    target_position = input_string.index('[')
    input_len = len(input_string)
    output_string = input_string[target_position:input_len] + input_string[0:target_position]
    return output_string


def accuracy(out, labels):
    outputs = np.argmax(out, axis=-1)
    return np.sum(outputs == labels)


class ShannonNER(object):
    def __init__(self, data_dir, bert_model_dir, fine_tuning_model_dir):
        self.max_seq_length = 128
        task_name = "MSRANER"
        eval_batch_size = 32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        processor = DataProcessor(os.path.join(data_dir, task_name), do_lower_case=True)
        processor.get_train_examples()
        self.label_list = processor.all_labels
        num_labels = len(self.label_list)

        self.tokenizer = BertTokenizer.from_pretrained(bert_model_dir, do_lower_case=True)

        output_model_file = os.path.join(fine_tuning_model_dir, task_name, "pytorch_model.bin")

        # Load a trained model that you have fine-tuned
        model_state_dict = torch.load(output_model_file)
        self.model = BertForTokenClassification.from_pretrained(bert_model_dir, state_dict=model_state_dict,
                                                                num_labels=num_labels)
        self.model.to(self.device)
        self.model.eval()
        self.all_labels = processor.all_labels

    def ner(self, sentence):
        examples = wrap_a_sentence_to_examples(sentence)
        features = convert_examples_to_features(examples, self.label_list, self.max_seq_length, self.tokenizer)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

        input_ids = all_input_ids.to(self.device)
        input_mask = all_input_mask.to(self.device)
        segment_ids = all_segment_ids.to(self.device)
        label_ids = all_label_ids.to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids, segment_ids, input_mask)

        example = get_output_file(logits, input_ids, input_mask, label_ids, self.tokenizer.ids_to_tokens,
                                  self.all_labels)[0]
        return zip(example.text, example.predicts)


def get_output_file(logits, input_ids, input_mask, label_ids, token_vocab, label_vocab):
    logits = logits.detach().cpu().numpy()
    predicts = np.argmax(logits, axis=-1)
    input_ids = input_ids.cpu().numpy()
    input_mask = input_mask.cpu().numpy()
    label_ids = label_ids.cpu().numpy()
    examples = []
    for sent_num in range(len(logits)):
        sentence_length = np.sum(input_mask[sent_num])
        input_tokens = [token_vocab[token] for token in input_ids[sent_num]][1: sentence_length - 1]
        labels = [label_vocab[token] for token in label_ids[sent_num]][1: sentence_length - 1]
        predictions = [label_vocab[token] for token in predicts[sent_num]][1: sentence_length - 1]
        examples.append(InputExample(text=input_tokens, labels=labels, predicts=predictions))
    return examples


if __name__ == "__main__":
    interactive()
