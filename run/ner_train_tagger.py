import os, jieba, csv
import time
import jieba.posseg as pseg

############################################
# 借助 jieba 自定义分词标记工具，实现数据标注
############################################

# sym: 症状， dis：疾病名称
biaoji = set(['sym', 'dis'])

# 标点符号集合：作为一个序列的结尾
fuhao = set(['。', '?', '？', '!', '！'])

data_path = "data/unstructured"

# 爬取的和来自结构化数据的 疾病名称、症状名称，第一列为名称，第二列为 'sym', 或者 'dis'
dics = csv.reader(open("./data/crawled_sym_dis_dict.csv", 'r', encoding='utf8'))


# 利用jieba自定义分词，进行专有名词输入
# 将识别对象加入jieba识别词表，标记视为词性
for row in dics:
    if len(row) == 2:
        jieba.add_word(row[0].strip(), tag=row[1].strip())
        # 强制加入词为一个joined整体
        jieba.suggest_freq(row[0].strip())


# =======================================================================================
# 读取目标文件，进行IOB格式的标记，并写入dev、train、test文件
start_time = time.time()

# 保存结果的文件
train = open(os.path.join("data/", "train.txt"), 'w', encoding='utf8')

for file in os.listdir(data_path):
    if ".txt" in file:
        fp = open(os.path.join(data_path, file), 'r', encoding='utf8')
        for line in fp:
            words = pseg.cut(line)    # 带词性切词
            # key: word； value： part of speech
            for key, value in words:
                if value.strip() and key.strip():
                    if value.strip() not in biaoji:
                        value = 'O'
                        # 按字标记
                        for achar in key.strip():
                            if achar and achar.strip() in fuhao:
                                string = achar + " " + value.strip() + "\n" + "\n"
                                train.write(string)
                            elif achar.strip() and achar.strip() not in fuhao:
                                string = achar + " " + value.strip() + "\n"
                                train.write(string)

                    elif value.strip() in biaoji:
                        begin = 0
                        for char in key.strip():
                            # 开始字，以 B- 开头
                            if begin == 0:
                                begin += 1
                                string1 = char + ' ' + 'B-' + value.strip() + '\n'

                                train.write(string1)
                            else:   # 开始字之后，以 I- 开头
                                string1 = char + ' ' + 'I-' + value.strip() + '\n'
                                train.write(string1)

                    else:
                        continue

# 处理后文件储存格式为：
# 患 O
# 者 O
# 以 O
# 腰 O
# 痛 O
# 伴 O
# 双 O
# 下 O
# 肢 O
# 疼 B-sym
# 痛 I-sym
# 半 O
# 年 O

end_time = time.time()
print(("IOB tagging used time is {} s. \n".format(end_time - start_time)))