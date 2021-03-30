import re
import pandas as pd
import os

def repair_book(pred, test_dir, max_len=20):
    reg = re.compile(r'(《.*》)')
    # 什么是bad_case,这些bad_case是如何产生的，如何找到这些bad_case
    temp = pred[pred.Category == 'book'][pred[pred.Category == 'book']
        .Privacy.apply(lambda x: re.findall(reg, x)).apply(lambda x: len(x) != 1)]
    # print('temp\n', temp)
    bad_data_book = temp[temp.Privacy.apply(lambda x: ('《' in x or '》' in x))]
    ret = {'ID': [], 'Category': [], 'Pos_b': [], 'Pos_e': [], 'Privacy': []}
    for idx, category, pos_b, pos_e, privacy in bad_data_book.values:
        if '《' in privacy:
            # 向后搜索 max_len 个字符
            with open(os.path.join(test_dir, str(idx) + '.txt'), 'r', encoding='utf-8') as f:
                content = f.readline()
            for i in range(1, min(len(content) - pos_e - 1, max_len)):
                index = pos_e + i
                if content[index] in (',.。！!《'):
                    break
                if content[index] == '》':
                    pos_e = index
                    break
            ret['ID'].append(idx)
            ret['Category'].append(category)
            ret['Pos_b'].append(pos_b)
            ret['Pos_e'].append(pos_e)
            ret['Privacy'].append(content[pos_b:pos_e + 1])
        if '》' in privacy:
            # 向前搜索max_len个字符
            with open(os.path.join(test_dir, str(idx) + '.txt'), 'r', encoding='utf-8') as f:
                content = f.readline()
            for i in range(1, min(pos_b, max_len)):
                index = pos_b - i - 1
                if content[index] in (',.。！!《'):
                    break
                if content[index] == '》':
                    pos_b = index
                    break
            ret['ID'].append(idx)
            ret['Category'].append(category)
            ret['Pos_b'].append(pos_b)
            ret['Pos_e'].append(pos_e)
            ret['Privacy'].append(content[pos_b:pos_e + 1])
    return ret

def address_qq(pred, test_dir):
    reg = re.compile(r'([^0-9])')
    temp = pred[pred.Category == 'QQ'][pred[pred.Category == 'QQ']
        .Privacy.apply(lambda x: re.findall(reg, x))]
    # temp = pred[pred.Category == 'book'][pred[pred.Category == 'book'].Privacy.apply(lambda x:len(x)>1)]

    # temp = pred[pred.Category == 'QQ'][pred[pred.Category == 'QQ']
    #     .Privacy.apply(lambda x: re.findall(reg, x))]
    print(temp)

def test():
    content = '成都原高新区管委会副主任挪用公款被判无期标准列日vs桑普多利亚本届GDC将邀请任天堂社长岩田聪与《合金装备》系列之父小岛秀夫登台进行主题演讲，小时候我们看绿野仙踪时，总会沉浸于绿野仙踪所描绘的童话世界中，憧憬着自己能像多罗茜一样，一位商铺的老板告诉记者：“办一个pos刷卡机，需要交1%的手续费。'
    pos_e = 43  # 976,position,40,43,天堂社长
    start = len(content) - pos_e - 1
    print(start)
    for i in range(1, min(start, 20)):
        index = pos_e + i
        if content[index] in (',.。!！《'):
            break
        if content[index] == '》':
            pos_e = index
            break
    print(pos_e)

if __name__ == '__main__':
    test_dir = r'G:\jupyter Notebook\ccf_隐私_deepshare\ccf-隐私-deepshare-baseline\DBC_code\data\test_data'
    pd_pred = pd.read_csv(r'C:\Users\wvbx\Downloads\predict20201124_BIOES.csv', sep=',')
    address_qq(pd_pred, test_dir)
    # ret = repair_book(pd_pred, test_dir)
    # pd_bad_case = pd.DataFrame(ret)
    # pd_bad_case.to_csv('./bad_case.csv', sep='\t', index=False)