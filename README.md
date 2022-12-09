# AML Project
After cloning, make sure to run the following commands:
```
cd VEQA
python -m venv env
source env/bin/activate
pip install -r requirements.txt
mkdir Trained
```

## Download Data
Run `./download.sh`
Note: This script downloads the COCO features, which take ~3 hours to download.

After running this script, you `data` directory should look as follows:
```
Add
Tree
Output
Here
```

Note: To reproduce VEQA in the SNLI-VE configuration, download the data using [this script](https://github.com/ChenRocks/UNITER/blob/master/scripts/download_ve.sh).

<!-- VQA Questions data [[link](https://visualqa.org/vqa_v1_download.html)] -->


## Training
Note: Change values in `config.json`, especially `base_dir`, `[train|eval]_hypothesis_path` and `output_dir`

Make sure to move `json` files generated from the hypothesis generator to `data/`

### Training
`python train.py --config config/config.json --mode train`

### Inference
`python train.py --config config/config.json --mode test`
