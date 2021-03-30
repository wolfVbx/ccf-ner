import codecs
import glob
import os
import shutil
from sklearn.model_selection import train_test_split

data_dir = r'G:\jupyter Notebook\ner\train'
# 划分验证集训练集
file_list = []
for file_name in os.listdir(data_dir):
    if file_name.endswith('.txt'):
        file_list.append(os.path.join(data_dir, file_name))

train_filelist, val_filelist = train_test_split(file_list, test_size=0.2, random_state=666)

def _cut(sentence):
    """
    将一段文本切分成多个句子
    :param sentence:
    :return:
    """
    new_sentence = []
    sen = []
    for i in sentence:
        if i in ['。', '！', '？', '?'] and len(sen) != 0: # 以'。', '！', '？', '?'作为一句话的结尾进行断句，并且这些标点不能在开头
            sen.append(i)
            new_sentence.append("".join(sen))
            sen = []
        else:
            sen.append(i)

    if len(new_sentence) <= 1:  # 一句话超过max_seq_length且没有句号的，用","分割，再长的不考虑了。
        new_sentence = []
        sen = []
        for i in sentence:
            if i.split(' ')[0] in ['，', ','] and len(sen) != 0:
                sen.append(i)
                new_sentence.append("".join(sen))
                sen = []
            else:
                sen.append(i)
    if len(sen) > 0:  # 若最后一句话无结尾标点，则加入这句话
        new_sentence.append("".join(sen))
    return new_sentence

def cut_test_set(text_list, len_treshold):
    cut_text_list = []
    cut_index_list = []
    for text in text_list:

        temp_cut_text_list = []
        text_agg = ''
        if len(text) < len_treshold:
            temp_cut_text_list.append(text)
        else:
            sentence_list = _cut(text)  # 一条数据被切分成多句话
            for sentence in sentence_list:
                if len(text_agg) + len(sentence) < len_treshold:
                    text_agg += sentence
                else:
                    temp_cut_text_list.append(text_agg)
                    text_agg = sentence
            temp_cut_text_list.append(text_agg)  # 加上最后一个句子

        cut_index_list.append(len(temp_cut_text_list))
        cut_text_list += temp_cut_text_list

    return cut_text_list, cut_index_list

# 设置样本长度
text_length = 250

def from_ann2dic(r_ann_path, r_txt_path, w_path, w_file):
    """数据处理"""
    q_dic = {}
    with codecs.open(r_ann_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip("\r\n")
            line_arr = line.split('\t')
            entityinfo = line_arr[1]
            entityinfo = entityinfo.split(' ')
            cls = entityinfo[0]
            start_index = int(entityinfo[1])
            end_index = int(entityinfo[2])
            length = end_index - start_index
            for r in range(length):
                if r == 0:
                    q_dic[start_index] = ("B-%s" % cls)
                else:
                    q_dic[start_index + r] = ("I-%s" % cls)

    with codecs.open(r_txt_path, "r", encoding="utf-8") as f:
        content_str = f.read()

    cut_text_list, cut_index_list = cut_test_set([content_str], text_length)

    i = 0
    for idx, line in enumerate(cut_text_list):
        w_path_ = "%s/%s-%s-new.txt" % (w_path, w_file, idx)
        with codecs.open(w_path_, "w", encoding="utf-8") as w:
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

def gen_train_set():
    # data_dir = r'./'
    for file in train_filelist:
        file = file.replace('\\', '/')
        #     print('file_re:',file)
        if file.find(".ann") == -1 and file.find(".txt") == -1:
            continue
        file_name = file.split('/')[-1].split('.')[0]
        #     print('file_name:',file_name)
        #     print('file:',file)
        r_ann_path = os.path.join(data_dir, "%s.ann" % file_name)
        r_txt_path = os.path.join(data_dir, "%s.txt" % file_name)
        #     print('r_ann_path:',r_ann_path)
        #     print('r_txt_path:',r_txt_path)
        w_path = './train_new/'
        w_file = file_name
        from_ann2dic(r_ann_path, r_txt_path, w_path, w_file)

def gen_val_set():
    for file in val_filelist:
        file = file.replace('\\', '/')
        if file.find(".ann") == -1 and file.find(".txt") == -1:
            continue
        file_name = file.split('/')[-1].split('.')[0]
        #     print('file:',file)
        #     print('file_name:',file_name)
        r_ann_path = os.path.join(data_dir, "%s.ann" % file_name)
        r_txt_path = os.path.join(data_dir, "%s.txt" % file_name)
        w_path = './val_new/'
        w_file = file_name
        from_ann2dic(r_ann_path, r_txt_path, w_path, w_file)

def merge_trainset():
    w_path = "./data/train.txt"
    for file in os.listdir('./train_new/'):
        path = os.path.join("./train_new", file)
        if not file.endswith(".txt"):
            continue
        q_list = []
        print("开始读取文件:%s" % file)
        with codecs.open(path, "r", encoding="utf-8") as f:
            line = f.readline()
            line = line.strip("\r\n")
            while line != "END O":
                q_list.append(line)
                line = f.readline()
                line = line.strip("\r\n")
        print("开始写入文本%s" % w_path)
        with codecs.open(w_path, "a", encoding="utf-8") as f:
            for item in q_list:
                if item.__contains__('\ufeff'):
                    print("===============")
                f.write('%s\n' % item)
            f.write('\n')
        f.close()

def merge_valset():
    w_path = "./data/val.txt"
    for file in os.listdir('./val_new/'):
        path = os.path.join("./val_new", file)
        if not file.endswith(".txt"):
            continue
        q_list = []

        with codecs.open(path, "r", encoding="utf-8") as f:
            line = f.readline()
            line = line.strip("\r\n")
            while line != "END O":
                q_list.append(line)
                line = f.readline()
                line = line.strip("\r\n")

        with codecs.open(w_path, "a", encoding="utf-8") as f:
            for item in q_list:
                if item.__contains__('\ufeff'):
                    print("===============")
                f.write('%s\n' % item)
            f.write('\n')
        f.close()

# 原始验证集拷贝
def copy_val_set():
    for file in val_filelist:
        file = file.replace('\\', '/')
        print('file:', file)
        file_name = file.split('/')[-1].split('.')[0]
        r_ann_path = os.path.join("./train", "%s.ann" % file_name)
        shutil.copy(file, "./val_data")
        shutil.copy(r_ann_path, "./val_data")

if __name__ == '__main__':
    gen_train_set()