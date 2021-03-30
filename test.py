from bert4keras.snippets import sequence_padding, DataGenerator
import os
import json
from tqdm import tqdm
import pandas as pd
import numpy as np
import re
from matplotlib import pyplot as plt
import glob

def read_txt(filename, use_line=True):
    """
    读取 txt 数据
    filename : str
    use_line : bool
    return   : list
    """
    with open(filename, 'r', encoding='utf8') as f:
        if use_line:
            ret = f.readlines()
        else:
            ret = f.read()
    return ret

def cut_sent(txt, symbol, max_len=250):
    """
    将一段文本切分成多个句子
    txt     : str
    symbol  : list e.g ['。', '！', '？', '?']
    max_len : int
    return  : list
    """
    new_sentence = []
    sen = []
    # 使用 symbol 对文本进行切分
    for i in txt:
        if i in symbol and len(sen) != 0:
            if len(sen) <= max_len:
                sen.append(i)
                new_sentence.append(''.join(sen))
                sen = []
                continue
            # 对于超过 max_len 的句子，使用逗号进行切分
            else:
                sen.append(i)
                tmp = ''.join(sen).split('，')
                for j in tmp[:-1]:
                    j += '，'
                    new_sentence.append(j)
                new_sentence.append(tmp[-1])
                sen = []
                continue
        sen.append(i)

    # 如果最后一个 sen 没有 symbol ，则加入切分的句子中。
    if len(sen) > 0:
        # 对于超过 max_len 的句子，使用逗号进行切分
        if len(sen) <= max_len:
            new_sentence.append(''.join(sen))
        else:
            tmp = ''.join(sen).split('，')
            for j in tmp[:-1]:
                j += '，'
                new_sentence.append(j)
            new_sentence.append(tmp[-1])
    return new_sentence

def agg_sent(text_list, symbol, max_len, treshold):
    """
    将文本切分成句子，然后将尽量多的合在一起，如果小于 treshold 就不拆分
    text_list : list
    symbol    : list e.g ['。', '！', '？', '?']
    max_len  : int
    treshold  : int
    return    : list, list
    """
    cut_text_list = []
    cut_index_list = []
    for text in text_list:

        temp_cut_text_list = []
        text_agg = ''
        # 如果没超过 treshold ，则将文本放入
        if len(text) < treshold:
            temp_cut_text_list.append(text)
        else:
            # 将一段文本切分成多个句子
            sentence_list = cut_sent(text, symbol, max_len)
            # 尽量多的把句子合在一起
            for sentence in sentence_list:
                if len(text_agg) + len(sentence) <= treshold:
                    text_agg += sentence
                else:
                    temp_cut_text_list.append(text_agg)
                    text_agg = sentence
            # 加上最后一个句子
            temp_cut_text_list.append(text_agg)

        cut_index_list.append(len(temp_cut_text_list))
        cut_text_list += temp_cut_text_list

    return cut_text_list, cut_index_list

def gen_data(label_path, data_path, output_path, output_file, symbol, max_len,
             treshold):
    """
    label_path : 标签路径
    data_path  : 数据路径
    output_path: 输出路径
    output_file: 输出文件
    symbol     : list e.g ['。', '！', '？', '?']
    max_len    : int
    treshold   : int
    """
    q_dic = {}
    tmp = pd.read_csv(label_path).values
    for _, entity_cls, start_index, end_index, entity_name in tmp:
        start_index = int(start_index)
        end_index = int(end_index)
        length = end_index - start_index + 1
        for r in range(length):
            if r == 0:
                q_dic[start_index] = ("B-%s" % entity_cls)
            else:
                q_dic[start_index + r] = ("I-%s" % entity_cls)

    content_str = read_txt(data_path)
    cut_text_list, cut_index_list = agg_sent(content_str, symbol, max_len,
                                             treshold)
    i = 0
    for idx, line in enumerate(cut_text_list):
        output_path_ = "%s/%s-%s-new.txt" % (output_path, output_file, idx)
        with open(output_path_, "w", encoding="utf-8") as w:
            for str_ in line:
                if str_ is " " or str_ == "" or str_ == "\n" or str_ == "\r":
                    pass
                else:
                    if i in q_dic:
                        tag = q_dic[i]
                    else:
                        tag = "O"  # 大写字母O
                    w.write('%s %s\n' % (str_, tag))
                i += 1
            w.write('%s\n' % "END O")

def gen_data_BIOES(label_path, data_path, output_path, output_file, symbol, max_len,
                   treshold):
    """
        label_path : 标签路径
        data_path  : 数据路径
        output_path: 输出路径
        output_file: 输出文件
        symbol     : list e.g ['。', '！', '？', '?']
        max_len    : int
        treshold   : int
        """
    q_dic = {}
    tmp = pd.read_csv(label_path).values
    for ID, entity_cls, start_index, end_index, entity_name in tmp:
        start_index = int(start_index)
        end_index = int(end_index)
        if start_index == end_index:
            q_dic[start_index] = ("S-%s" % entity_cls)
        elif end_index - start_index == 1:
            q_dic[start_index] = ("B-%s" % entity_cls)
            q_dic[end_index] = ("E-%s" % entity_cls)
        else:
            try:
                q_dic[start_index] = ("B-%s" % entity_cls)
                q_dic[end_index] = ("B-%s" % entity_cls)
                for pos_i in range(start_index + 1, end_index):
                    q_dic[pos_i] = ("I-%s" % entity_cls)
            except:
                print('无法标注的样本：', ID)

    content_str = read_txt(data_path)
    cut_text_list, cut_index_list = agg_sent(content_str, symbol, max_len,
                                             treshold)
    i = 0
    for idx, line in enumerate(cut_text_list):
        output_path_ = "%s/%s-%s-new.txt" % (output_path, output_file, idx)
        with open(output_path_, "w", encoding="utf-8") as w:
            for str_ in line:
                if str_ is " " or str_ == "" or str_ == "\n" or str_ == "\r":
                    pass
                else:
                    if i in q_dic:
                        tag = q_dic[i]
                    else:
                        tag = "O"  # 大写字母O
                    w.write('%s %s\n' % (str_, tag))
                i += 1
            w.write('%s\n' % "END O")

def load_data(filename):
    """
    加载生成的数据
    filename : str
    return   : list
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        f = f.read()
        # 按照 '\n\n' 获取数据数据（聚合的句子）
        for l in f.split('\n\n'):
            if not l:
                continue
            d, last_flag = [], ''
            for c in l.split('\n'):
                try:
                    char, this_flag = c.split(' ')
                except:
                    print('Exception:{}end'.format(c))
                    continue
                if this_flag == 'O' and last_flag == 'O':
                    d[-1][0] += char
                elif this_flag == 'O' and last_flag != 'O':
                    d.append([char, 'O'])
                elif this_flag[:1] == 'B':
                    d.append([char, this_flag[2:]])
                else:
                    try:
                        d[-1][0] += char
                    except:
                        print(l)
                        print(d)
                        continue
                last_flag = this_flag
            D.append(d)
    return D

def checkout(filename):
    """
    检查提交数据
    filename : str
    """
    all_lines = read_txt(filename)
    for line in all_lines:
        if not line.split('\n')[-1] == '':
            print(line)
        else:
            if len((line.split('\n')[0]).split(',')) != 5:
                print(line)

def data_analysis_ner():
    # 把train 和 dev 的数据全部用来训练，需要把数据处理成
    # j.keys(): ['text', 'label'],text:训练数据集，label:标签
    labels = list()
    sentences = []
    with open('./cluener_public/train.json', 'r', encoding='utf-8') as f:
        for line in f:
            j_line = json.loads(line)
            text = j_line['text']
            sentences.append(text)
            label_dict = j_line['label']
            for k, v in label_dict.items():
                labels.append(k)
        pd_label = pd.DataFrame(labels, columns=['category'])
        # 各个类别标签分布情况
        print(pd.value_counts(pd_label['category']))
        # 句子长度分布情况
        len_list = []
        for sen in sentences:
            len_list.append(len(sen))
        plt.hist(len_list, bins=20)

def json_to_dataFrame(ext_out_path):
    with open('./cluener_public/labeled_test.json', 'r', encoding='utf-8') as f_in:
        for i, line in enumerate(f_in):
            label_list = list()
            j_line = json.loads(line)
            # text = j_line['text']
            label_dict = j_line['label']
            for label, entity_dict in label_dict.items():
                # "label": {"game": {"CSOL": [[4, 7]]}}
                # 先把其改装成pd.DataFrame
                for entity, pos_list in entity_dict.items():
                    # entity: 布莱克本 ,pos_list: [[14, 17], [30, 33]] 2
                    # print('entity:', entity, ',pos_list:', pos_list, len(pos_list))
                    for pos in pos_list:
                        # pandas 数据结构：category,pos_b,pos_e,entity
                        label_list.append([i, label, pos[0], pos[1], entity])
            pd_label = pd.DataFrame(label_list, columns=['ID', 'Category', 'Pos_b', 'Pos_e', 'Privacy'])
            pd_label.to_csv(ext_out_path + str(i + 1) + '.csv', sep='\t', index=False)

def data_to_text(ext_path):
    '''json中的文本提出出来'''
    with open('./cluener_public/dev.json', 'r', encoding='utf-8') as f_in:
        for i, line in enumerate(f_in):
            with open(ext_path + str(i + 1) + '.txt', 'w', encoding='utf-8') as fout:
                j_line = json.loads(line)
                text = j_line['text']
                fout.write(text)

def gen_BIO_data():
    output_path = 'G:\jupyter Notebook\ccf_隐私_deepshare\ccf-隐私-deepshare-baseline\DBC_code\data\ext_data\ext_train_bio\\'
    label_dir = 'G:\jupyter Notebook\ccf_隐私_deepshare\ccf-隐私-deepshare-baseline\DBC_code\data\ext_data\ext_label\*.csv'
    train_data_dir = 'G:\jupyter Notebook\ccf_隐私_deepshare\ccf-隐私-deepshare-baseline\DBC_code\data\ext_data\ext_train_data\\'
    label_list = glob.glob(label_dir)
    for _, label_path in enumerate(label_list):
        label_index = label_path[label_path.rindex('\\') + 1:].split('.')[0]
        bio_dict = dict()
        data = pd.read_csv(label_path, sep='\t').values
        for _, entity_cls, start_idx, end_idx, entity in data:
            start_idx = int(start_idx)
            end_idx = int(end_idx)
            label_len = end_idx - start_idx + 1
            for offset in range(label_len):
                if offset == 0:
                    bio_dict[start_idx] = 'B-%s' % entity_cls
                else:
                    bio_dict[start_idx + offset] = 'I-%s' % entity_cls
        with open(train_data_dir + str(label_index) + '.txt', 'r', encoding='utf-8') as f_in:
            output_name = output_path + str(label_index) + '.txt'
            with open(output_name, 'w', encoding='utf-8') as f_out:
                line = f_in.readlines()
                index = 0
                for word in list(line[0]):  # 只有一句话，所以不用担心别的问题
                    if word is ' ' or word == '' or word == '\n' or word == '\t':
                        continue
                    else:
                        if index in bio_dict:
                            tag = bio_dict[index]
                        else:
                            tag = 'O'
                        f_out.write(word + ' ' + tag + '\n')
                    index += 1
                f_out.write('%s\n' % "END O")

def gen_BIOES_data(output_path, label_dir, train_data_dir):
    label_list = glob.glob(label_dir)
    for label_path in label_list:
        # label_index = label_path[label_path.rindex('\\') + 1:].split('.')[0]
        # aug-0-14-1-new.txt
        # label_index = os.path.basename(label_path).split('-')[2]
        basename = os.path.basename(label_path)
        bioes_dict = dict()
        data = pd.read_csv(label_path, sep=',').values
        for _, entity_cls, start_idx, end_idx, entity in data:
            start_idx = int(start_idx)
            end_idx = int(end_idx)
            if start_idx == end_idx:
                bioes_dict[start_idx] = 'S-%s' % entity_cls
            elif end_idx - start_idx == 1:
                bioes_dict[start_idx] = 'B-%s' % entity_cls
                bioes_dict[end_idx] = 'E-%s' % entity_cls
            else:
                bioes_dict[start_idx] = 'B-%s' % entity_cls
                bioes_dict[end_idx] = 'E-%s' % entity_cls
                for pos_i in range(start_idx + 1, end_idx):
                    bioes_dict[pos_i] = 'I-%s' % entity_cls
        with open(train_data_dir + basename.split('.')[0] + '.txt', 'r', encoding='utf-8') as f_in:
            # output_name = output_path + basename.split('.')[0] + '.txt'
            output_name = os.path.join(output_path, basename.split('.')[0] + '.txt')
            with open(output_name, 'w', encoding='utf-8') as f_out:
                line = f_in.readlines()
                index = 0
                for word in list(line[0]):  # 只有一句话，所以不用担心别的问题
                    if word is ' ' or word == '' or word == '\n' or word == '\t':
                        continue
                    else:
                        if index in bioes_dict:
                            tag = bioes_dict[index]
                        else:
                            tag = 'O'
                        f_out.write(word + '\t' + tag + '\n')
                    index += 1
                # f_out.write('%s\n' % "END O")

def labeled_json_to_pd(ext_out_path):
    with open('./cluener_public/labeled_test.json', 'r', encoding='utf-8') as f_in:
        for i, line in enumerate(f_in):
            label_list = list()
            j_line = json.loads(line)
            entities = j_line['entities']
            index = j_line['id']
            text = j_line['text']
            prob_value_list = j_line['prob_value'][0][1:-1]
            tag_seq = j_line['tag_seq']
            for entity in entities:
                label = entity[0]
                pos_s = entity[1]
                pos_e = entity[2]
                entity_ = text[int(pos_s):int(pos_e) + 1]
                prob = sum(prob_value_list[int(pos_s):int(pos_e) + 1]) / len(entity_)
                label_list.append([index, label, pos_s, pos_e, entity_, prob])
            pd_label = pd.DataFrame(label_list, columns=['ID', 'Category', 'Pos_b', 'Pos_e', 'Privacy', 'Prob'])
            pd_label.to_csv(ext_out_path + str(index) + '.csv', sep='\t', index=False)

def get_label_from_aug_data(in_path_dir, out_path_label, output_data_path):
    files = glob.glob(in_path_dir)
    for file in files:
        basename = os.path.basename(file)
        # 数据增强的下标 # aug-0-14-1-new.txt 原来的文件名称
        file_index = basename.split('-')[2]
        txt_list, label_list = [], []
        # 记录行下标，来计算实体的start,end角标
        start_index, end_index = 0, 0
        pd_labels = {'ID': [], 'Category': [], 'Pos_b': [], 'Pos_e': [], 'Privacy': []}
        with open(file, 'r', encoding='utf-8') as f_in:
            lines = f_in.readlines()
            for line in lines:
                char, candidate_label = line.split('\t')
                candidate_label = candidate_label.replace('\n', '')
                # 需要拿到两部分字符和标签
                txt_list.append(char)
                label_list.append(candidate_label)
            left_search_index = 0
            for _ in label_list:
                # txt_list和label_list中的元素是一一对应的
                if left_search_index >= len(label_list):
                    break
                if label_list[left_search_index] == 'O':
                    left_search_index += 1
                    continue
                if label_list[left_search_index].startswith('B'):
                    # B 确定start_index开始的位置，并标记开始找到实体，因此为了逻辑上的完整，
                    # 需要添加 starting = True
                    entity_label = label_list[left_search_index].split('-')[1]
                    start_index = left_search_index
                    # 相对start_index的偏移量
                    offset = 0
                    # 不能一上来就是O,同时还是需要处理两个实体连在一起的情况
                    starting = True
                for j in range(start_index + 1, len(label_list)):
                    if label_list[j].startswith('I'):
                        offset += 1
                        starting = True
                        continue
                    elif starting and label_list[j].startswith('O'):
                        # 一个实体结束后，后面就是O,即没有出现两个实体连在一起
                        end_index = start_index + offset + 1
                        entity_name = ''.join(txt_list[start_index:end_index])
                        pd_labels['ID'].append(file_index)
                        pd_labels['Category'].append(entity_label)
                        pd_labels['Pos_b'].append(start_index)
                        pd_labels['Pos_e'].append(end_index - 1)
                        pd_labels['Privacy'].append(entity_name)
                        left_search_index = end_index
                        starting = False
                        break
                    elif starting and label_list[j].startswith('B'):
                        # 这里需要考虑两个实体连在一起的情况
                        end_index = start_index + offset + 1
                        entity_name = ''.join(txt_list[start_index:end_index])
                        pd_labels['ID'].append(file_index)
                        pd_labels['Category'].append(entity_label)
                        pd_labels['Pos_b'].append(start_index)
                        pd_labels['Pos_e'].append(end_index - 1)
                        pd_labels['Privacy'].append(entity_name)
                        starting = False
                        left_search_index = end_index
                        break
        pd_result = pd.DataFrame(pd_labels)
        pd_result.to_csv(os.path.join(out_path_label, basename.split('.')[0] + '.csv'), index=False)
        with open(os.path.join(output_data_path, basename.split('.')[0] + '.txt'), 'w',
                  encoding='utf-8') as f_out:
            f_out.write(''.join(txt_list))

if __name__ == '__main__':
    # path_in_dir = r'G:\jupyter Notebook\ccf_隐私_deepshare\ccf-隐私-deepshare-ch7\data\train_aug2\*.txt'
    # out_path_label = r'G:\jupyter Notebook\ccf_隐私_deepshare\ccf-隐私-deepshare-baseline\DBC_code\data\train_aug\train_aug2\label'
    # output_data_path = r'G:\jupyter Notebook\ccf_隐私_deepshare\ccf-隐私-deepshare-baseline\DBC_code\data\train_aug\train_aug2\data'
    # get_label_from_aug_data(path_in_dir, out_path_label, output_data_path)

    # data_analysis_ner()
    # output_path = './cluener_for_ccf.txt'
    ext_path = r'G:\jupyter Notebook\ccf_隐私_deepshare\ccf-隐私-deepshare-baseline\DBC_code\data\ext_data\ext_test_label\\'
    # labeled_json_to_pd(ext_path)
    # ext_path = r'G:\jupyter Notebook\ccf_隐私_deepshare\ccf-隐私-deepshare-baseline\DBC_code\data\ext_data\ext_test_data\\'
    # test_data_to_text(ext_path)
    # data_to_text(ext_path)
    # gen_BIO_data()
    # bad_case_analyse(path, label)
    # output_path = 'G:\jupyter Notebook\ccf_隐私_deepshare\ccf-隐私-deepshare-baseline\DBC_code\data\ext_data\ext_test_bioes\\'
    # label_dir = 'G:\jupyter Notebook\ccf_隐私_deepshare\ccf-隐私-deepshare-baseline\DBC_code\data\ext_data\ext_test_label\*.csv'
    # train_data_dir = 'G:\jupyter Notebook\ccf_隐私_deepshare\ccf-隐私-deepshare-baseline\DBC_code\data\ext_data\ext_test_data\\'

    output_path = r'G:\jupyter Notebook\ccf_隐私_deepshare\ccf-隐私-deepshare-baseline\DBC_code\data\train_aug\train_aug2_bioes\train_bioes\\'
    label_dir = r'G:\jupyter Notebook\ccf_隐私_deepshare\ccf-隐私-deepshare-baseline\DBC_code\data\train_aug\train_aug2\label\*.csv'
    train_data_dir = r'G:\jupyter Notebook\ccf_隐私_deepshare\ccf-隐私-deepshare-baseline\DBC_code\data\train_aug\train_aug2\data\\'
    gen_BIOES_data(output_path, label_dir, train_data_dir)
