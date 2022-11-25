import json

questionsFile = open('data/MultipleChoice_mscoco_train2014_questions.json')
annotationFile = open('data/Annotations_Train_mscoco/mscoco_train2014_annotations.json')

data = json.load(questionsFile)
annotationData = json.load(annotationFile)

for annotationIterator, annotationItem in enumerate(annotationData['annotations']):
    for iterator, item in enumerate(data['questions']):
        if annotationItem['question_id'] == item['question_id']:
            data['questions'][iterator]['question_type'] = annotationItem['question_type']
            break

with open('data.json', 'w') as output:
    json.dump(data, output, ensure_ascii=False)