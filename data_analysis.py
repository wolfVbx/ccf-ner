import os
import numpy as np
import pandas as pd
import codecs
import matplotlib.pyplot as plt

"""
通过数据分析存在的问题：
1. 样本量少，一共才1000个
2. 样本标签分布及其不均衡
3. 文本长度长，而且文本长度变化大，最短的几十个字，最长的1400+，大多数分布在[100,800]之间，偏向于100
4. 文本对应的领域专业性强，对任务理解会产生一定的影响
"""

data_dir = r'G:\jupyter Notebook\ner\train'

def get_content_info():
    content_len = []
    count = 0
    for file in os.listdir(data_dir):
        if file.endswith('.txt'):
            with open(os.path.join(data_dir, file), 'r', encoding='utf-8') as fin:
                content_str = fin.read()
                content_len.append(len(content_str))
                # 通过查看文档，发现文档长度长短不一，需要看看文档长短分布
                # print(content_str)
                # print('-'*30)
                # count += 1
                # if count == 5:
                #     break
    print('样本量：', len(content_len))
    # 查看文本分布长度
    plt.rcParams['font.family'] = 'SimHei'
    plt.rcParams['font.size'] = 20
    plt.figure(figsize=(10, 5))
    plt.hist(content_len, bins=20, facecolor="blue", edgecolor="black", range=(0, 1500))
    plt.xlabel('文本长度')
    plt.ylabel('样本数')
    plt.title('文本长度分布直方图')
    plt.show()

def get_label_info():
    count = 0
    df_label = pd.DataFrame()
    for file in os.listdir(data_dir):
        if file.endswith('.ann'):
            data = pd.read_csv(os.path.join(data_dir, file), sep='\t', names=['id', 'entityInfo', 'entity'])
            data['category'] = data['entityInfo'].apply(lambda x: x.split(' ')[0])
            data['offset1'] = data['entityInfo'].apply(lambda x: x.split(' ')[1]).astype(int)
            data['offset2'] = data['entityInfo'].apply(lambda x: x.split(' ')[2]).astype(int)
            data = data[['id', 'entity', 'category', 'offset1', 'offset2']]
            df_label = pd.concat([df_label, data])
            # print(df_label.groupby('category').count().reset_index()[['category', 'id']]
            #       .sort_values(by='id', ascending=False))
            print(df_label.groupby('entity').count().reset_index()[['entity', 'id']]
                  .sort_values(by='id', ascending=False))
            # print(df_label)

if __name__ == '__main__':
    # get_label_info()
    get_content_info()