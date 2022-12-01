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


import json
import os
import base64

import numpy as np
import pandas as pd
import torch

from torch import FloatTensor

from torch.utils.data import Dataset

import lmdb
import msgpack
import msgpack_numpy


from collections import defaultdict
from contextlib import contextmanager
import io
import json
from os.path import exists

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset

from tqdm import tqdm
import lmdb
# from lz4.frame import compress, decompress

import msgpack
import msgpack_numpy

def decode_numpy(obj, chain=None):
    """
    Decoder for deserializing numpy data types.
    """

    try:
        if b"nd" in obj:
            if obj[b"nd"] is True:

                # Check if b'kind' is in obj to enable decoding of data
                # serialized with older versions (#20):
                if b"kind" in obj and obj[b"kind"] == b"V":
                    descr = [
                        tuple(tostr(t) if type(t) is bytes else t for t in d)
                        for d in obj[b"type"]
                    ]
                else:
                    descr = obj[b"type"]
                return np.frombuffer(obj[b"data"], dtype=np.dtype(descr)).reshape(
                    obj[b"shape"]
                )
            else:
                descr = obj[b"type"]
                return np.frombuffer(obj[b"data"], dtype=np.dtype(descr))[0]
        elif b"complex" in obj:
            return complex(tostr(obj[b"data"]))
        else:
            return obj if chain is None else chain(obj)
    except KeyError:
        return obj if chain is None else chain(obj)


class DetectFeatLmdb(object):
    def __init__(self, img_dir, conf_th=0.2, max_bb=100, min_bb=10, num_bb=36,
                 compress=False):
        self.img_dir = img_dir
        if conf_th == -1:
            db_name = f'feat_numbb{num_bb}'
            self.name2nbb = defaultdict(lambda: num_bb)
        else:
            db_name = f'feat_th{conf_th}_max{max_bb}_min{min_bb}'
            nbb = f'nbb_th{conf_th}_max{max_bb}_min{min_bb}.json'
            if not exists(f'{img_dir}/{nbb}'):
                # nbb is not pre-computed
                self.name2nbb = None
            else:
                self.name2nbb = json.load(open(f'{img_dir}/{nbb}'))
        self.compress = compress
        if compress:
            db_name += '_compressed'

        if self.name2nbb is None:
            if compress:
                db_name = 'all_compressed'
            else:
                db_name = 'all'

        # only read ahead on single node training
        self.env = lmdb.open(f'{img_dir}/{db_name}',
                             readonly=True, create=False,
                             readahead=not False)
        self.txn = self.env.begin(buffers=True)
        
        if self.name2nbb is None:
            self.name2nbb = self._compute_nbb()

    def _compute_nbb(self):
        name2nbb = {}
        fnames = json.loads(self.txn.get(key=b'__keys__').decode('utf-8'))
        for fname in tqdm(fnames, desc='reading images'):
            dump = self.txn.get(fname.encode('utf-8'))
            if self.compress:
                with io.BytesIO(dump) as reader:
                    img_dump = np.load(reader, allow_pickle=True)
                    confs = img_dump['conf']
            else:
                img_dump = msgpack.loads(dump, raw=False)
                confs = img_dump['conf']
            name2nbb[fname] = compute_num_bb(confs, self.conf_th,
                                             self.min_bb, self.max_bb)

        return name2nbb

    def __del__(self):
        self.env.close()

    def get_dump(self, file_name):
        # hack for MRC
        dump = self.txn.get(file_name.encode('utf-8'))
        nbb = self.name2nbb[file_name]
        if self.compress:
            with io.BytesIO(dump) as reader:
                img_dump = np.load(reader, allow_pickle=True)
                img_dump = _fp16_to_fp32(img_dump)
        else:
            img_dump = msgpack.loads(dump, raw=False)
            img_dump = _fp16_to_fp32(img_dump)
        img_dump = {k: arr[:nbb, ...] for k, arr in img_dump.items()}
        return img_dump

    def __getitem__(self, file_name):
        dump = self.txn.get(file_name.encode('utf-8'))
        nbb = self.name2nbb[file_name]
        if self.compress:
            with io.BytesIO(dump) as reader:
                img_dump = np.load(reader, allow_pickle=True)
                img_dump = {'features': img_dump['features'],
                            'norm_bb': img_dump['norm_bb']}
        else:
            img_dump = msgpack.loads(dump, raw=False)
        
        img_feat = torch.tensor(decode_numpy(img_dump['features'])[:nbb, :]).float()
        img_bb = torch.tensor(decode_numpy(img_dump['norm_bb'])[:nbb, :]).float()
        breakpoint()
        return img_feat, img_bb



class VEQADataset(Dataset):
    def __init__(self, mode):
        super(VEQADataset, self).__init__()
        # os.path.join(base_dir, feature_path)

        self.ans2label = {"contradiction": 0, "entailment": 1}
        self.label2ans = {0: "contradiction", 1: "entailment"}

        FLICKR30KDB = "/home/meghana/meg/e-ViL/data/esnlive/img_db/flickr30k/feat_th0.2_max100_min10"
        FLICKR30KDB_NBB = "/home/meghana/meg/e-ViL/data/esnlive/img_db/flickr30k/nbb_th0.2_max100_min10.json"
        TEXT = "/home/meghana/meg/e-ViL/data/esnlive_train.csv" if mode=="train" else "/home/meghana/meg/e-ViL/data/esnlive_dev.csv"

        img_path = FLICKR30KDB
        nbb_path = FLICKR30KDB_NBB
        text_path = TEXT

        self.env = lmdb.open(
            img_path, readonly=True, create=False, readahead=not False
        )
        self.txn = self.env.begin(buffers=True)
        self.name2nbb = json.load(open(nbb_path))
        self.annotations = pd.read_csv(text_path)

        self.annotations = json.loads(self.annotations.to_json(orient="records"))
        self.annotations = list(filter(lambda x: x["gold_label"]!="neutral", self.annotations))

        # self.features = zarr.open(os.path.join(base_dir, feature_path), mode='r')
        # self.boxes = zarr.open(os.path.join(base_dir, boxes_path), mode='r')

        # self.datapoints = json.load(open(os.path.join(base_dir, questions_path)))["questions"]
        # self.annotations = json.load(open(os.path.join(base_dir, annos_path)))["annotations"]
        
        # self.annotations = dict(zip(list(d["question_id"] for d in self.annotations), self.annotations))
        
        self.tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
        
        self.max_len = 64 #args.get("max_length", 100)
        
        # datapoints = self.datapoints
        # data_length = len(datapoints)
        
        # for i in tqdm(range(data_length)):
        #     qid = self.datapoints[i]["question_id"]
        #     assert qid in self.annotations
        #     self.datapoints[i]["q_type"] = self.annotations[qid]["question_type"]
        #     self.datapoints[i]["answer"] = self.annotations[qid]["multiple_choice_answer"]

        # self.num_ans = args.get("num_ans", len(self.datapoints[0]["multiple_choices"]))
        # self.__process_vqa()

    def __process_vqa(self):
        for ind, datapoint in enumerate(self.datapoints):
            q = datapoint["question"]

            #Contains our hypotheses, which are merged question answer sentences.
            qas = []

            #Contains our entialment scores between 0-1 for each hypothesis. for now 1.0 if answer. 0 otherwise.
            scores = []

            answer = self.datapoints[ind]["answer"]
            a_ind = self.datapoints[ind]["multiple_choices"].index(answer)
            assert a_ind>=0

            # Selecting `self.num_ans` (set to 4 for now) number of answer choies from given 18. 
            answers = sample(self.datapoints[ind]["multiple_choices"][:a_ind]+self.datapoints[ind]["multiple_choices"][a_ind+1:], self.num_ans-1)
            answers+=[answer]

            # Needs to be completed
            for a in answers:

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
                hypothesis = q+" "+a

                # ========== =========================== ==========
                
                qas.append(hypothesis)
            
            assert len(qas) == len(scores)
            self.datapoints[ind]["qas"], self.datapoints[ind]["scores"] = qas, scores

    def __getitem__(self, index):
        item = self.annotations[index]
        flickr30k_ID = item["Flickr30kID"]
        i = "flickr30k_%s"%flickr30k_ID.replace(".jpg", ".npz").zfill(16)


        dump = self.txn.get(i.encode("utf-8"))
        nbb = self.name2nbb[i]

        img_dump = msgpack.loads(dump, raw=False)
        feats = decode_numpy(img_dump["features"])[:nbb, :]
        img_bb = decode_numpy(img_dump["norm_bb"])[:nbb, :4]

        # get box to same format than used by code's authors
        # boxes = np.zeros((img_bb.shape[0], 7), dtype="float32")
        # boxes[:, :-1] = img_bb[:, :]
        # boxes[:, 4] = img_bb[:, 5]
        # boxes[:, 5] = img_bb[:, 4]
        # boxes[:, 4] = img_bb[:, 5]
        # boxes[:, 6] = boxes[:, 4] * boxes[:, 5]




#         # breakpoint()

#         # print(len(max(item["qas"], key=lambda x:len(x)).split()))
        tokenized_question = self.tokenizer(item["hypothesis"], 
                                            add_special_tokens=True,   # Adds [CLS] and [SEP] token to every input text
                                            max_length=self.max_len, 
                                            return_tensors="pt",
                                            truncation=True, 
                                            padding="max_length")
        
#         tokens_padded = tokenized_question["input_ids"]
#         # for tokenized_qa in tokenized_question["input_ids"]:
#         #     if len(tokenized_qa)>self.max_length:
#         #         tokens_padded.append(tokenized_qa[:self.max_length])
#         #     else:
#         #         tokens_padded.append(tokenized_qa + [self.tokenizer('[PAD]')['input_ids'][1:-1][0]]*(self.max_length - len(tokenized_qa)))
        
#         # torch.LongTensor(tokens_padded),
        
#         return item["image_id"], item["question_id"], item["q_type"], item["answer"], tokenized_question, torch.from_numpy(np.array(self.features[item["image_id"]])),torch.from_numpy(np.array(self.boxes[item["image_id"]])), torch.FloatTensor(item["scores"])#,tokenized_question["attention_mask"],tokenized_question["token_type_ids"]
        # #             best_eval_score = eval_score
        return FloatTensor(feats.copy()), FloatTensor(img_bb.copy()), tokenized_question, self.ans2label[item["gold_label"]]
    
    def __len__(self):
        return len(self.annotations)

        

if __name__=="__main__":
    v = VEQADataset() # DetectFeatLmdb("/home/meghana/meg/e-ViL/data/esnlive/img_db/flickr30k")
    c = v.__getitem__(4)
    print(c)
    # breakpoint()
    # v.__getitem__("flickr30k_005225747391.npz")

        
    
