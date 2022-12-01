# AML_Project
`cd VEQA`\
`python -m venv env`\
`source env/source/activate`\
`pip install -r requirements.txt`\
`mkdir Trained`\


## Download Data
`./download.sh`

VQA Questions data [[link](https://visualqa.org/vqa_v1_download.html)]


## Training
Note: Change values in `config.json`, especially `base_dir` and `output_dir`


`python train.py --config config/config.json --mode train`