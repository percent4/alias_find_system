# -*- coding: utf-8 -*-
# @Time : 2021/9/5 23:54
# @Author : Jclian91
# @File : model_predict.py
# @Place : Yangpu, Shanghai
# -*- coding: utf-8 -*-

import json
import numpy as np
from bert4keras.backend import K, batch_gather
from bert4keras.layers import Loss
from bert4keras.layers import LayerNormalization
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_exponential_moving_average
from bert4keras.snippets import open, to_array
from keras.layers import Input, Dense, Lambda, Reshape
from keras.models import Model


maxlen = 200
config_path = './chinese-RoBERTa-wwm-ext/bert_config.json'
checkpoint_path = './chinese-RoBERTa-wwm-ext/bert_model.ckpt'
dict_path = './chinese-RoBERTa-wwm-ext/vocab.txt'

predicate2id, id2predicate = {}, {}

with open('./data/alias_schema', "r", encoding="utf-8") as f:
    for l in f:
        l = json.loads(l)
        if l['predicate'] not in predicate2id:
            id2predicate[len(predicate2id)] = l['predicate']
            predicate2id[l['predicate']] = len(predicate2id)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 补充输入
subject_labels = Input(shape=(None, 2), name='Subject-Labels')
subject_ids = Input(shape=(2,), name='Subject-Ids')
object_labels = Input(shape=(None, len(predicate2id), 2), name='Object-Labels')

# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    return_keras_model=False,
)

# 预测subject
output = Dense(
    units=2, activation='sigmoid', kernel_initializer=bert.initializer
)(bert.model.output)
subject_preds = Lambda(lambda x: x**2)(output)

subject_model = Model(bert.model.inputs, subject_preds)


# 根据subject_ids从output中取出subject的向量表征
def extract_subject(inputs):
    output, subject_ids = inputs
    start = batch_gather(output, subject_ids[:, :1])
    end = batch_gather(output, subject_ids[:, 1:])
    subject = K.concatenate([start, end], 2)
    return subject[:, 0]


# 传入subject，预测object
# 通过Conditional Layer Normalization将subject融入到object的预测中
output = bert.model.layers[-2].get_output_at(-1)
subject = Lambda(extract_subject)([output, subject_ids])
output = LayerNormalization(conditional=True)([output, subject])
output = Dense(
    units=len(predicate2id) * 2,
    activation='sigmoid',
    kernel_initializer=bert.initializer
)(output)
output = Lambda(lambda x: x**4)(output)
object_preds = Reshape((-1, len(predicate2id), 2))(output)

object_model = Model(bert.model.inputs + [subject_ids], object_preds)


class TotalLoss(Loss):
    """subject_loss与object_loss之和，都是二分类交叉熵
    """
    def compute_loss(self, inputs, mask=None):
        subject_labels, object_labels = inputs[:2]
        subject_preds, object_preds, _ = inputs[2:]
        if mask[4] is None:
            mask = 1.0
        else:
            mask = K.cast(mask[4], K.floatx())
        # sujuect部分loss
        subject_loss = K.binary_crossentropy(subject_labels, subject_preds)
        subject_loss = K.mean(subject_loss, 2)
        subject_loss = K.sum(subject_loss * mask) / K.sum(mask)
        # object部分loss
        object_loss = K.binary_crossentropy(object_labels, object_preds)
        object_loss = K.sum(K.mean(object_loss, 3), 2)
        object_loss = K.sum(object_loss * mask) / K.sum(mask)
        # 总的loss
        return subject_loss + object_loss


subject_preds, object_preds = TotalLoss([2, 3])([
    subject_labels, object_labels, subject_preds, object_preds,
    bert.model.output
])

# 训练模型
train_model = Model(
    bert.model.inputs + [subject_labels, subject_ids, object_labels],
    [subject_preds, object_preds]
)

AdamEMA = extend_with_exponential_moving_average(Adam, name='AdamEMA')
optimizer = AdamEMA(lr=1e-5)
train_model.compile(optimizer=optimizer)


train_model.load_weights('./models/alias_best_model.weights')


# 抽取输入text所包含的三元组
def extract_spoes(text):

    tokens = tokenizer.tokenize(text, maxlen=maxlen)
    mapping = tokenizer.rematch(text, tokens)
    token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
    token_ids, segment_ids = to_array([token_ids], [segment_ids])
    # 抽取subject
    subject_preds = subject_model.predict([token_ids, segment_ids])
    start = np.where(subject_preds[0, :, 0] > 0.6)[0]
    end = np.where(subject_preds[0, :, 1] > 0.5)[0]
    subjects = []
    for i in start:
        j = end[end >= i]
        if len(j) > 0:
            j = j[0]
            subjects.append((i, j))
    if subjects:
        spoes = []
        token_ids = np.repeat(token_ids, len(subjects), 0)
        segment_ids = np.repeat(segment_ids, len(subjects), 0)
        subjects = np.array(subjects)
        # 传入subject，抽取object和predicate
        object_preds = object_model.predict([token_ids, segment_ids, subjects])
        for subject, object_pred in zip(subjects, object_preds):
            start = np.where(object_pred[:, :, 0] > 0.6)
            end = np.where(object_pred[:, :, 1] > 0.5)
            for _start, predicate1 in zip(*start):
                for _end, predicate2 in zip(*end):
                    if _start <= _end and predicate1 == predicate2:
                        spoes.append(
                            ((mapping[subject[0]][0],
                              mapping[subject[1]][-1]), predicate1,
                             (mapping[_start][0], mapping[_end][-1]))
                        )
                        break
        return [(text[s[0]:s[1] + 1], id2predicate[p], text[o[0]:o[1] + 1])
                for s, p, o, in spoes]
    else:
        return []


if __name__ == '__main__':
    new_text = "机场三字代码简称“三字代码”,由国际航空运输协会(IATA ,International Air Transport Association)制定、用于代表全世界大多数机场的代码。"
    print(extract_spoes(new_text))