# Download instruction
Download dataset from [[here](https://visualqa.org/vqa_v1_download.html)]

Store dataset in data folder.

`python preprocess_questiontypes.py`\
`python sentence_generator.py`

The code stores finalData.json which has keys as ['question', 'sentences', 'image_id', 'question_id', 'question_type'].

There are 65 question types. We have made templates for each of them. The templates can be found in `anstype.csv`. Sentences are generated using those templates.

For processing the finalData.zip file:

```
import zipfile 

with zipfile.ZipFile("VEQA/data_preprocess/processed data/finalData.zip","r") as zip_ref:
    zip_ref.extractall("VEQA/data_preprocess/processed data/")

```
This will return finalData.json 