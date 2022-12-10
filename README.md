# AML Project

Note: All the steps listed below are enclosed in `VEQA/run.sh`. Execute `source ./run.sh` to get the code set up and running.

Note: If running the script shows `ERROR 429: Too Many Requests.`, it could unfortuantely be wget running into a "multiple request" issue. In this case, please follow these steps: 
1. Download the data from [here](https://drive.google.com/file/d/1sk55WOI1UkeprcyNT4f6ppoLcjzSK1ua/view?usp=sharing) and place it under `VEQA/`. 
2. Also download the VEQA checkpoint from [here](https://drive.google.com/file/d/13PvcN-Ebn_s57Q-1C6EqKEIvUcExRn9a/view) and place it under `VEQA/Trained`
3. Run `source ./run.sh`. It would still show the `ERROR 429: Too Many Requests.`, but would continue unzipping the files sine there are now present.

## Running Instructions

### Set up the Environment
After cloning, make sure to run the following commands (make sure you are in the root directory `VEQA/`):
```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
mkdir Trained
```

### Download the Data
Run `./download.sh` to download the image features and question-answer annotations.

Note: This script downloads the COCO features, which take ~3 hours to download. To expedite things, we provide an abdridged version of the data for quick running/testing of code here. It contain ~5% of the total number of questions (amounting to ~12,000 questions). Although please note that we trained our models on the full data.

After running this script, you `data` directory should look as follows:
```
coco/
evalfinalData.json
finalData.json
mscoco_train2014_annotations.json
mscoco_val2014_annotations.json
MultipleChoice_mscoco_train2014_questions.json
MultipleChoice_mscoco_val2014_questions.json
```

Note: To reproduce VEQA in the SNLI-VE configuration, download the data using [this script](https://github.com/ChenRocks/UNITER/blob/master/scripts/download_ve.sh).

<!-- VQA Questions data [[link](https://visualqa.org/vqa_v1_download.html)] -->


### Training
Note: Change values in `config.json`, especially `base_dir`, `[train|eval]_hypothesis_path` and `output_dir`

#### Training
`python run.py --config config/config.json --mode train`

#### Inference
`python run.py --config config/config.json --mode test --model <model path>`

#### VEQA(SNLI-VE) Configuration
`ve_train.py` and `ve_dataset.py` contain the model configuration for SNLI-VE training and the dataloader for SNLI-VE respectively. `ve_train.py` can be run the same was as `run.py`.


#### Files
`run.py` - VEQA train/test pipeline for AS_{VQA} configuration\
`model.py` - VEQA model code\
`ve_train.py` - VEQA train pipeline for AS_{SNLI-VE} configuration\
`data_preprocess/sentence_generator.py` - Code for precomputing hypothesis\
`dataset.py` - Dataset for VQA dataset\
`ve_dataset.py` - Dataset for SNLI-VE dataset

