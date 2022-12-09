import pandas as pd
import json

ansdata = pd.read_csv('anstype.csv')

keys = ansdata['questiontype'].to_list()
values = ansdata['prompt'].to_list()
questionPrompt = dict(zip(keys, values))

def sentence_generator(question_type, multiple_choices):
    sentenceList = []
    prompt = questionPrompt[question_type]
    for choice in multiple_choices:
        sentenceList.append(prompt + ' ' + choice)
    return sentenceList

preprocessedFile = open('../data/MultipleChoice_mscoco_train2014_questions.json')
data = json.load(preprocessedFile)

finalData = []
for i, item in enumerate(data['questions']):
    dicts = {}
    keys = ['question', 'sentences', 'image_id', 'question_id', 'question_type']
    dicts['question'] = data['questions'][i]['question']
    dicts['sentences'] = sentence_generator(data['questions'][i]['question_type'], data['questions'][i]['multiple_choices'])
    dicts['image_id'] = data['questions'][i]['image_id']
    dicts['question_id'] = data['questions'][i]['question_id']
    dicts['question_type'] = data['questions'][i]['question_type']
    finalData.append(dicts)

with open('FinalData.json', 'w') as outfile:
  json.dump(finalData, outfile, ensure_ascii=False)