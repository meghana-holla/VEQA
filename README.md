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
Note: Change values in `config.json`, especially `base_dir`, `[train|eval]_hypothesis_path` and `output_dir`

Make sure to move `json` files generated from the hypothesis generator to `data/`

`python train.py --config config/config.json --mode train`
