import os
import pickle
import tqdm
import ahocorasick


def build_symptom_ac(input_path, out_file):
    "构建 所有疾病症状的 AC 自动机"
    AC = ahocorasick.Automaton()

    idx = 0
    for symlist in tqdm.tqdm(load_data(input_path)):
        for key in symlist:
            if key not in AC:
                AC.add_word(key, (idx, key))

    AC.make_automaton()
    with open(out_file, "wb") as f:
        pickle.dump(AC, f)

    print("Builded AC.")


def load_data(path):
    csv_list = os.listdir(path)
    # disease_names = list(map(lambda x: x.split(".")[0], csv_list))

    for csv in csv_list:
        with open(os.path.join(path, csv), 'r', encoding='utf-8') as f:
            symptom = list(map(lambda x: x.strip(), f.readlines()))
        symptom = list(filter(lambda x: 0 < len(x) < 100, symptom))

        yield symptom


if __name__ == '__main__':
    build_symptom_ac("data/structured/reviewed", "weights/AC.bin")