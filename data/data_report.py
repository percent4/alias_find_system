# -*- coding: utf-8 -*-
# @Time : 2021/9/2 13:46
# @Author : Jclian91
# @File : data_report.py
# @Place : Yangpu, Shanghai
import json
from random import shuffle

with open("ccf2019_corpus.json", "r", encoding="utf-8") as f:
    content = json.loads(f.read())

with open("sougouqa_webqa_corpus.json", "r", encoding="utf-8") as f:
    content.extend([_ for _ in json.loads(f.read()) if _["spo"][0][0]])

# 简单数据统计
print(f"共有{len(content)}条标注样本")
print("data review for last 10 samples: ")
for example in content[-10:]:
    print(example)

# 将数据集划分为训练集和测试集，比列为8:2
shuffle(content)
train_data = content[:int(len(content)*0.8)]
test_data = content[int(len(content)*0.8):]
with open("alias_train.json", "w", encoding="utf-8") as f:
    for _ in train_data:
        f.write(json.dumps(_, ensure_ascii=False)+"\n")
with open("alias_test.json", "w", encoding="utf-8") as f:
    for _ in test_data:
        f.write(json.dumps(_, ensure_ascii=False)+"\n")

