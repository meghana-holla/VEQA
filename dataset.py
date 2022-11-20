import json
import os
import zarr
from torch.utils.data import Dataset

class VEQADataset(Dataset):
    def __init__(self, split, base_dir, questions_path, feature_path, boxes_path):
        
        self.features = zarr.open(os.path.join(base_dir, feature_path), mode='r')
        self.boxes = zarr.open(os.path.join(base_dir, boxes_path), mode='r')
        self.datapoints = process_vqa(base_dir, questions_path)

        self.tokenizer = None


        for i in json.load(open(os.path.join(base_dir, questions_path)))["questions"]:
            breakpoint()

    def __process_vqa(self, base_dir, questions_path):
        results = {}
        for i in json.load(open(os.path.join(base_dir, questions_path)))["questions"]:
            breakpoint()
        

if __name__=="__main__":
    d = VEQADataset("train", "/home/meghana/meg/VEQA/data", "MultipleChoice_mscoco_train2014_questions.json", "", "")
