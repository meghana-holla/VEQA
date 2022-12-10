echo "Setting up the environment"
python -m venv env
source env/bin/activate
pip install -r requirements.txt
mkdir Trained

echo "Downloading abridged VEQA data"
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1sk55WOI1UkeprcyNT4f6ppoLcjzSK1ua' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1sk55WOI1UkeprcyNT4f6ppoLcjzSK1ua" -O veqadata.tar.gz && rm -rf /tmp/cookies.txt
tar -xvf veqadata.tar.gz
rm veqadata.tar.gz

echo "Downloading pretrained VEQA model for HQ+VQA configuration"
cd ../Trained
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=13PvcN-Ebn_s57Q-1C6EqKEIvUcExRn9a' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=13PvcN-Ebn_s57Q-1C6EqKEIvUcExRn9a" -O VEQA_best_model.pth && rm -rf /tmp/cookies.txt
cd ..

echo "Running VEQA inference branch"
python run.py --config config/config.json --mode test --model Trained/VEQA_best_model.pth


