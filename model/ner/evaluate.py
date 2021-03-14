


# 构建函数评估模型的准确率和召回率, F1值
def evaluate(sentence_list, gold_tags, predict_tags, id2char, id2tag):
    """计算准确率，召回率, F1值

    params：
        sentence_list: text
        gold_tags: 真实的标签序列
        predict_tags: 模型预测出来的标签序列
        id2char: id to char
        id2tag: id to tag
    """

    if predict_tags.ndim > 2:
        predict_tags = predict_tags.squeeze(0)

    tp_count, pred_count, gold_count = 0, 0, 0
    for sent_idx, sentence in enumerate(sentence_list):
        gold_entities, gold_entity = [], []
        predict_entities, predict_entity = [], []
        for char_idx in range(len(sentence)):
            # <PAD>: 在 使用 pack_padded 方法条件下，是忽略了 <PAD> 位置的，所以可以不考虑
            if sentence[char_idx] == 0:
                break

            char_text = id2char[sentence[char_idx].item()]
            gold_tag_type = id2tag[gold_tags[sent_idx][char_idx].item()]
            predict_tag_type = id2tag[predict_tags[sent_idx][char_idx].item()]

            # 首先判断真实实体是否可以加入列表中
            # 首先判断id2tag的第一个字符是否为B, 表示一个实体的开始
            if gold_tag_type[0] == "B":
                gold_entity = [char_text + "/" + gold_tag_type]

            # 是否为I, 表示一个实体的中间到结尾
            # 总体的判断条件:1.类型要以I开始 2.entity不为空 3.实体类型相同
            elif gold_tag_type[0] == "I" and len(gold_entity) != 0 \
                  and gold_entity[-1].split("/")[1][1:] == gold_tag_type[1:]:
                gold_entity.append(char_text + "/" + gold_tag_type)

            # 是否为O, 并且entity非空, 实体已经完成了全部的判断
            elif gold_tag_type[0] == "O" and len(gold_entity) != 0:
                gold_entity.append(str(sent_idx) + "_" + str(char_idx))  # 唯一标识
                gold_entities.append(gold_entity)
                gold_entity = []
            else:
                gold_entity = []

            # 判断预测出来的命名实体
            # 实体的开始
            if predict_tag_type[0] == "B":
                predict_entity = [char_text + "/" + predict_tag_type]

            # 是否是I, 并且entity非空, 并且实体类型相同
            elif predict_tag_type[0] == "I" and len(predict_entity) != 0 \
                 and predict_entity[-1].split("/")[1][1:] == predict_tag_type[1:]:
                predict_entity.append(char_text + "/" + predict_tag_type)

            # 是否为O, 并且entity非空, 代表一个完整的实体已经识别完毕, 可以追加进列表中
            elif predict_tag_type[0] == "O" and len(predict_entity) != 0:
                predict_entity.append(str(sent_idx) + "_" + str(char_idx))  # 唯一标识
                predict_entities.append(predict_entity)
                predict_entity = []
            else:
                predict_entity = []

        acc_entities = [entity for entity in predict_entities if entity in gold_entities]
        # 计算正确预测出来的实体个数
        tp_count += len(acc_entities)
        # 计算预测了多少个实体
        pred_count += len(predict_entities)
        # 计算真实实体的个数
        gold_count += len(gold_entities)

    # 计算准确率,召回率, F1值
    if tp_count > 0:
        acc = float(tp_count / pred_count)
        recall = float(tp_count / gold_count)
        f1_score = 2.0 * acc * recall / (acc + recall)
        return acc, recall, f1_score, tp_count, pred_count, gold_count
    else:
        return 0, 0, 0, tp_count, pred_count, gold_count
