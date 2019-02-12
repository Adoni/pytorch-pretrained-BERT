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
from shannon_ner.shannon_ner import ShannonNER


def interactive():
    shannon_ner = ShannonNER(data_dir="/data/nfsdata/nlp/datasets/sequence_labeling/CN_NER",
                             bert_model_dir="/data/nfsdata/nlp/BERT_BASE_DIR/chinese_L-12_H-768_A-12/",
                             fine_tuning_model_dir="/data/nfsdata/data/sunxiaofei/checkpoints/ner")

    while 1:
        sentence = input("Sentence: ")
        named_entities = shannon_ner.ner(sentence)
        for token, pred in named_entities:
            print(f"{token} {pred}")


if __name__ == "__main__":
    interactive()
