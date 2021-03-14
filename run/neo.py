import os
import tqdm
from neo4j import GraphDatabase

CONFIG = {
    "uri": "bolt://127.0.0.1:7687",
    "auth": ("neo4j", "123456"),
    "encrypted": False
}

driver = GraphDatabase.driver(**CONFIG)


def load_data(path):
    csv_list = os.listdir(path)
    disease_names = list(map(lambda x: x.split(".")[0], csv_list))

    symptom_list = []
    for csv in csv_list:
        with open(os.path.join(path, csv), 'r', encoding='utf-8') as f:
            symptom = list(map(lambda x: x.strip(), f.readlines()))
        symptom = list(filter(lambda x: 0 < len(x) < 100, symptom))
        symptom_list.append(symptom)

    return dict(zip(disease_names, symptom_list))


def write(path):
    "写入neo4j"
    disease_and_symptom = load_data(path)

    with driver.session() as sess:
        for dis_name, symptom in tqdm.tqdm(disease_and_symptom.items()):
            add_dis = "MERGE (a:Disease{name:%r}) RETURN a" % dis_name
            sess.run(add_dis)
            for sym in symptom:
                add_sym = "MERGE (b:Symptom{name:%r}) RETURN b" % sym
                sess.run(add_sym)
                add_relation = "MATCH (a:Disease{name:%r}) MATCH (b:Symptom{name:%r}) \
								 WITH a,b MERGE (a)-[r:dis_to_sym]-(b)" % (dis_name, sym)
                sess.run(add_relation)

        add_idx1 = "CREATE INDEX ON:Disease(name)"
        sess.run(add_idx1)
        add_idx2 = "CREATE INDEX ON:Symptom(name)"
        sess.run(add_idx2)


if __name__ == '__main__':
    # 处理结构化的数据
    # 处理之前，可以先删掉空文件
    # find ./rawcsv -name "*.csv" -type f -size 0c | xargs -n 1 rm -f
    write("data/structured/reviewed")
