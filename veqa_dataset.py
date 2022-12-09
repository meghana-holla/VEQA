import json
import os
import zarr
from tqdm import tqdm
import torch
import numpy as np

from torch.utils.data import Dataset
from torch.optim import Adam
from transformers import LxmertTokenizer, LxmertModel
import utils

from random import sample

class VEQADataset(Dataset):
    def __init__(self, split, base_dir, questions_path, annos_path, feature_path, boxes_path, args):
        super(VEQADataset, self).__init__()
        # os.path.join(base_dir, feature_path)
        self.features = zarr.open(os.path.join(base_dir, feature_path), mode='r')
        self.boxes = zarr.open(os.path.join(base_dir, boxes_path), mode='r')
        self.datapoints = json.load(open(os.path.join(base_dir, questions_path)))["questions"]
        self.annotations = json.load(open(os.path.join(base_dir, annos_path)))["annotations"]
        #
        hpath="%s_hypothesis_path"%split
        self.hypothesis = json.load(open(os.path.join(base_dir, args[hpath])))
        
        self.annotations = dict(zip(list(d["question_id"] for d in self.annotations), self.annotations))
        
        self.tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
        
        self.max_len = args.get("max_length", 64)
        
        datapoints = self.datapoints
        data_length = len(datapoints)
        
        for i in tqdm(range(data_length)):
            qid = self.datapoints[i]["question_id"]
            assert qid in self.annotations
            self.datapoints[i]["q_type"] = self.annotations[qid]["question_type"]
            self.datapoints[i]["answer"] = self.annotations[qid]["multiple_choice_answer"]

        self.num_ans = args.get("num_ans", len(self.datapoints[0]["multiple_choices"]))

        self.hg_mode = args.get("hg_mode", 0)
        
        if self.hg_mode ==1:
            self.hypothesis_dicts = {}
            for d in self.hypothesis:
                self.hypothesis_dicts[d['question_id']] = d['sentences']

        self.__process_vqa()

    def __process_vqa(self):
        for ind, datapoint in enumerate(self.datapoints):
            
            q = datapoint["question"]
            qid = datapoint["question_id"]

            #Contains our hypotheses, which are merged question answer sentences.
            qas = []

            #Contains our entialment scores between 0-1 for each hypothesis. for now 1.0 if answer. 0 otherwise.
            scores = []

            answer = self.datapoints[ind]["answer"]
            a_ind = self.datapoints[ind]["multiple_choices"].index(answer)
            assert a_ind>=0

            # Selecting `self.num_ans` (set to 4 for now) number of answer choies from given 18. Here selecting the answer index
            answers_index = sample(list(range(a_ind))+list(range(a_ind+1, len(self.datapoints[ind]["multiple_choices"]))), self.num_ans-1) # sample(self.datapoints[ind]["multiple_choices"][:a_ind]+self.datapoints[ind]["multiple_choices"][a_ind+1:], self.num_ans-1)
            
            # Finding a rnadom position to insert our answer
            answer_position = sample(range(self.num_ans), 1)[0]

            
            
            # Inserting our answer index into answers_index at answer_position
            answers_index.insert(answer_position, a_ind)

            answers = [self.datapoints[ind]["multiple_choices"][i] for i in answers_index]

            if self.hg_mode ==1:
                hypothesis_answers = self.hypothesis_dicts[qid]
                qas = [hypothesis_answers[i] for i in answers_index]

            # Needs to be completed
            for _, a in enumerate(answers):

                if a.strip() == answer.strip():
                    scores.append(1)
                else:
                    scores.append(0)

                #Create the sentence for question (q) and answer (a) 
                
                # Eg: 
                # q: hat animal is in the pond? 
                # a: elephant 
                # Hypothesis: elephant is in the pond.
                
                # ========== HYPOTHESIS GENERATION LOGIC ==========
                
                # this needs to be replaced with our hypothesis generation logic.
                if self.hg_mode == 0:
                    # print("Hello")
                    hypothesis = q+" "+a
                    qas.append(hypothesis)

                # ========== =========================== ==========
                
                # qas.append(hypothesis)
            
            assert len(qas) == len(scores)

            self.datapoints[ind]["qas"], self.datapoints[ind]["scores"] = qas, scores

        

    def __getitem__(self, index):
        item = self.datapoints[index]
        # print(len(max(item["qas"], key=lambda x:len(x)).split()))
        tokenized_question = self.tokenizer(item["qas"], 
                                                add_special_tokens=True,   # Adds [CLS] and [SEP] token to every input text
                                                max_length=self.max_len, 
                                                return_tensors="pt",
                                                truncation=True, 
                                                padding="max_length")
        
        tokens_padded = tokenized_question["input_ids"]
        # for tokenized_qa in tokenized_question["input_ids"]:
        #     if len(tokenized_qa)>self.max_length:
        #         tokens_padded.append(tokenized_qa[:self.max_length])
        #     else:
        #         tokens_padded.append(tokenized_qa + [self.tokenizer('[PAD]')['input_ids'][1:-1][0]]*(self.max_length - len(tokenized_qa)))
        
        # torch.LongTensor(tokens_padded),
        
        return item["image_id"], item["question_id"], item["q_type"], item["answer"], tokenized_question, torch.from_numpy(np.array(self.features[item["image_id"]])),torch.from_numpy(np.array(self.boxes[item["image_id"]])), torch.FloatTensor(item["scores"])#,tokenized_question["attention_mask"],tokenized_question["token_type_ids"]
    
    def __len__(self):
        return len(self.datapoints)

if __name__=="__main__":
    d = VEQADataset("train", "/home/meghana/meg/VEQA/data", "MultipleChoice_mscoco_train2014_questions.json", "mscoco_train2014_annotations.json", "coco/trainval.zarr", "coco/trainval_boxes.zarr", args={"num_ans": 4})
    x = d.__getitem__(4)
    breakpoint()


        
    
