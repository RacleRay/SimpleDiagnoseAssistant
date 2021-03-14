import os
import tqdm
import sys
import json
import time


def gen_neg_entities(path, outpath):
    """
    构建实体审核模型，这一步标注数据来自人工审核，但是负样本标注直接用逆序字符串，这就很naive。
    人工审核不通过的，应该也记下来，作为负样本。也可以随机生成。
    但是，最稳定的做法，是构建一个数据字典之类的数据结构，直接在标准中查询是否存在。这不比使用
    RNN代价小得多吗，还稳定。
    所以 实体审核 模型，在这种实现模式下，有待商榷。
    """
    files = os.listdir(path)

    with open(outpath, "w", encoding='utf-8') as to:
        for name in tqdm.tqdm(files):
            with open(os.path.join(path, name), "r", encoding='utf-8') as f:
                for line in f.readlines():
                    if len(line) > 1:
                        to.write("0," + str(line[::-1]).strip() + "\n")
                        to.write("1," + str(line).strip() + "\n")



def char_id_map(input_path, out_file):
    word_to_ix = {"<PAD>": 0}

    files = os.listdir(input_path)
    for file in files:
        with open(os.path.join(input_path, file) "r", encoding='utf-8') as f:
            for line in f.readlines():
                for char in line:
                    if char not in word_to_ix:
                        word_to_ix[char] = len(word_to_ix)
    word_to_ix["UNK"] = len(word_to_ix)

    fp = open(out_file, 'w', encoding='utf-8')
    json.dump(word_to_ix, fp, ensure_ascii=False, indent=4)
    fp.flush()
    time.sleep(3)
    fp.close()


if __name__ == '__main__':
    sys.exit(gen_neg_entities("./data/structured/reviewed", "./data/structured/necheck_train.csv"))