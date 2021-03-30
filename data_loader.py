import json

def concat_json(json1_path, json2_path):
    ret = []
    with open(json1_path, 'r', encoding='utf-8') as j1_file:
        with open(json2_path, 'r', encoding='utf-8') as j2_file:
            j1_lines = j1_file.readlines()
            j2_lines = j2_file.readlines()
            for line1 in j1_lines:
                for line2 in j2_lines:
                    json1 = json.loads(line1)
                    json2 = json.loads(line2)
                    line1_id = json1['id']
                    line2_id = json2['id']
                    if int(line1_id) == int(line2_id):
                        result = dict()
                        result['id'] = line1_id
                        result['text'] = json1['text']
                        # print('text:', json1['text'])
                        result['tag_seq'] = json2['tag_seq']
                        result['entities'] = json2['entities']
                        result['prob_value'] = json2['prob_value']
                        ret.append(result)
                        continue
            with open('./cluener_public/labeled_test.json', 'w', encoding='utf-8') as f_out:
                for record in ret:
                    f_out.write(json.dumps(record, ensure_ascii=False)+'\n')


def read_json(path):
    with open(path, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()
        for line in lines:
            print(line)
            print('---------')

if __name__ == '__main__':
    json1_file = r'G:\pythonproject\work\drag_ner\cluener_public\test.json'
    json2_file = r'C:\Users\wvbx\Downloads\test_prediction.json'
    concat_json(json1_file, json2_file)
    path = r'G:\pythonproject\work\drag_ner\cluener_public\labeled_test.json'
    # read_json(path)