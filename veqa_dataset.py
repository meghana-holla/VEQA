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
    def __init__(self, split, base_dir, questions_path, annos_path, feature_path, boxes_path, args={"num_ans": 4}):
        super(VEQADataset, self).__init__()
        # os.path.join(base_dir, feature_path)
        self.features = zarr.open(os.path.join(base_dir, feature_path), mode='r')
        self.boxes = zarr.open(os.path.join(base_dir, boxes_path), mode='r')
        self.datapoints = json.load(open(os.path.join(base_dir, questions_path)))["questions"]
        self.annotations = json.load(open(os.path.join(base_dir, annos_path)))["annotations"]
        
        self.annotations = dict(zip(list(d["question_id"] for d in self.annotations), self.annotations))
        
        self.tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
        
        self.max_length = args.get("max_length", 25)
        
        datapoints = self.datapoints
        data_length = len(datapoints)
        
        for i in tqdm(range(data_length)):
            qid = self.datapoints[i]["question_id"]
            assert qid in self.annotations
            self.datapoints[i]["q_type"] = self.annotations[qid]["question_type"]
            self.datapoints[i]["answer"] = self.annotations[qid]["multiple_choice_answer"]
            # breakpoint()

        self.num_ans = args.get("num_ans", len(self.datapoints[0]["multiple_choices"]))
        self.__process_vqa()

        # for i in json.load(open(os.path.join(base_dir, questions_path)))["questions"]:
        #     breakpoint()

    def __process_vqa(self):
        for ind, datapoint in enumerate(self.datapoints):
            q = datapoint["question"]
            qas = []
            scores = []

            answer = self.datapoints[ind]["answer"]
            a_ind = self.datapoints[ind]["multiple_choices"].index(answer)
            assert a_ind>=0
            answers = sample(self.datapoints[ind]["multiple_choices"][:a_ind]+self.datapoints[ind]["multiple_choices"][a_ind+1:], self.num_ans-1)
            answers+=[answer]
            for a in answers:
                qas.append(q+" "+a)
                
                # Somewhere is will be our Hypothesis Generator Module.
                
                if a.strip() == answer.strip():
                    scores.append(1)
                else:
                    scores.append(0)
            assert len(qas) == len(scores)
            self.datapoints[ind]["qas"], self.datapoints[ind]["scores"] = qas, scores

    def __getitem__(self, index):
        item = self.datapoints[index]
        
        tokenized_question = self.tokenizer(item["qas"])
        tokens_padded = []
        for tokenized_qa in tokenized_question["input_ids"]:
            if len(tokenized_qa)>self.max_length:
                tokens_padded.append(tokenized_qa[:self.max_length])
            else:
                tokens_padded.append(tokenized_qa + [self.tokenizer('[PAD]')['input_ids'][1:-1][0]]*(self.max_length - len(tokenized_qa)))
        
        # {'input_ids': [101, 2003, 2023, 2028, 6847, 2030, 3674, 19571, 1029, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
        # return {
        #     "id": item["image_id"],
        #     "qid": item["question_id"],
        #     "q_type": item["q_type"],
        #     "answer": item["answer"],
        #     "qa": None,
        #     "inputs": {
        #         "visual_feats": torch.from_numpy(np.array(self.features[item["image_id"]])),
        #         "visual_pos": torch.from_numpy(np.array(self.boxes[item["image_id"]])),
        #         **tokenized_question,
        #     }
        # }
        return item["image_id"], item["question_id"], item["q_type"], item["answer"], torch.LongTensor(tokens_padded), torch.from_numpy(np.array(self.features[item["image_id"]])),torch.from_numpy(np.array(self.boxes[item["image_id"]])), torch.FloatTensor(item["scores"])#,tokenized_question["attention_mask"],tokenized_question["token_type_ids"]
    
    def __len__(self):
        return len(self.datapoints)

# def compute_score_with_logits(outputs, scores):
#     s = outputs*scores
#     return s

# def evaluate(model, loader, cfg):
#     model.eval()

#     eval_score = 0
#     total_loss = 0

#     for i, (imid, qid, q_type, answer, qa_tokens, features, boxes, scores) in tqdm(enumerate(loader)):
#                 qa_tokens = qa_tokens.cuda()
#                 features = features.cuda()
#                 boxes = boxes.cuda()
#                 scores = scores.cuda()

#                 qa_tokens = qa_tokens.reshape(qa_tokens.shape[0] * dataset.num_ans, -1)
#                 features_r = features.repeat(1, dataset.num_ans, 1)
#                 features = features_r.reshape(features.shape[0] * dataset.num_ans, features.shape[1], features.shape[2])
#                 boxes_r = boxes.repeat(1, dataset.num_ans , 1)
#                 boxes = boxes_r.reshape(boxes.shape[0] * dataset.num_ans,boxes.shape[1], boxes.shape[2])
                
#                 outputs = model(qa_tokens, features, boxes)
#                 outputs = outputs.reshape(-1, dataset.num_ans)

#                 loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, scores, reduction="mean") * scores.size(1)
#                 i+=1

#                 score = compute_score_with_logits(outputs, scores.data).sum()

#                 eval_score += score.item()
#                 total_loss += loss.item() * features.size(0)

#                 final_loss = total_loss / (i * features.size(0))
#                 final_score = 100 * eval_score / (i * features.size(0))

#     return final_loss, final_score
        

# if __name__=="__main__":
#     class AttributeDict(dict):
#         __getattr__ = dict.__getitem__
#         __setattr__ = dict.__setitem__
#         __delattr__ = dict.__delitem__

#     cfg = AttributeDict({"num_ans": 18, "epochs": 10, "batch_size": 4, "output_dir": "/home/meghana/meg/VEQA/Trained", "base_dir":"/home/meghana/meg/VEQA/data", "train_q": "MultipleChoice_mscoco_train2014_questions.json", "train_a": "mscoco_train2014_annotations.json", "features": "coco/trainval.zarr", "boxes": "coco/trainval_boxes.zarr"})

#     from torch.utils.data import DataLoader
#     from model import VEQ
#     dataset = VEQADataset("train", cfg.base_dir, cfg.train_q, cfg.train_a, cfg.features, cfg.boxes)
#     loader = DataLoader(dataset, cfg.batch_size, shuffle=True, num_workers=1, collate_fn=utils.trim_collate)
#     eval_loader = loader
#     aa = dataset.__getitem__(2)
    
#     model = VEQ(cfg)
#     model = model.cuda()

#     optim = Adam(model.parameters())

#     # lxmert = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
#     # tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
#     # inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
#     i=1
#     best_eval_score = 0
#     train_score = 0
#     total_loss = 0

    
#     for epoch in range(cfg.epochs):
#         print(f"Epoch{epoch}/{cfg.epochs}:\n")
#         model.train()
#         for i, (imid, qid, q_type, answer, qa_tokens, features, boxes, scores) in tqdm(enumerate(loader)):
#             qa_tokens = qa_tokens.cuda()
#             features = features.cuda()
#             boxes = boxes.cuda()
#             scores = scores.cuda()

#             qa_tokens = qa_tokens.reshape(qa_tokens.shape[0] * dataset.num_ans, -1)
#             features_r = features.repeat(1, dataset.num_ans, 1)
#             features = features_r.reshape(features.shape[0] * dataset.num_ans, features.shape[1], features.shape[2])
#             boxes_r = boxes.repeat(1, dataset.num_ans , 1)
#             boxes = boxes_r.reshape(boxes.shape[0] * dataset.num_ans,boxes.shape[1], boxes.shape[2])
            
#             outputs = model(qa_tokens, features, boxes)
#             outputs = outputs.reshape(-1, dataset.num_ans)

#             loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, scores, reduction="mean") * scores.size(1)
#             i+=1

#             loss.backward()
#             optim.step()
#             optim.zero_grad()


#             score = compute_score_with_logits(outputs, scores.data).sum()

#             train_score += score.item()
#             total_loss += loss.item() * features.size(0)

#             if i != 0 and i % 2 == 0:
#                 print(
#                     'training: %d/%d, train_loss: %.6f, train_acc: %.6f' %
#                     (i, len(loader), total_loss / (i * features.size(0)),
#                      100 * train_score / (i * features.size(0))))

#         total_loss /= len(loader)
#         model.eval()
#         eval_loss, eval_score = evaluate(model, eval_loader, cfg)
#         model.train()

#         # logger.write('\nlr: %.7f' % optim.param_groups[0]['lr'])
#         # logger.write('epoch %d, time: %.2f' % (epoch, time.time() - t))
#         # logger.write(
#         #     '\ttrain_loss: %.2f, norm: %.4f, score: %.2f' % (total_loss, total_norm / count_norm, train_score))
#         # if eval_loader is not None:
#         print('\teval score: %.2f' % (100 * eval_score))


#         if (eval_loader is not None and eval_score > best_eval_score):
#         #     if opt.lp == 0:
#             model_path = os.path.join(cfg.output_dir, "VEQA_best_model.pth")
#         #     elif opt.lp == 1:
#         #         model_path = os.path.join(opt.output, 'SAR_SSL_top'+str(opt.train_candi_ans_num)+'_best_model.pth')
#         #     elif opt.lp == 2:
#         #         model_path = os.path.join(opt.output, 'SAR_LMH_top'+str(opt.train_candi_ans_num)+'_best_model.pth')
#             utils.save_model(model_path, model, epoch, optim)
#             best_eval_score = eval_score



        
    
