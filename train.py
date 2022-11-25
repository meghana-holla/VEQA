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

from argparse import ArgumentParser

from veqa_dataset import VEQADataset


def compute_score_with_logits(outputs, scores):
    # s = outputs*scores
    s = torch.argmax(outputs, dim=1)==torch.argmax(scores, dim=1)
    return s>0

def evaluate(model, loader, cfg):
    model.eval()

    eval_score = 0
    total_loss = 0

    for i, (imid, qid, q_type, answer, qa_tokens, features, boxes, scores) in tqdm(enumerate(loader)):
                qa_tokens = qa_tokens.cuda()
                features = features.cuda()
                boxes = boxes.cuda()
                scores = scores.cuda()

                qa_tokens = qa_tokens.reshape(qa_tokens.shape[0] * dataset.num_ans, -1)
                features_r = features.repeat(1, dataset.num_ans, 1)
                features = features_r.reshape(features.shape[0] * dataset.num_ans, features.shape[1], features.shape[2])
                boxes_r = boxes.repeat(1, dataset.num_ans , 1)
                boxes = boxes_r.reshape(boxes.shape[0] * dataset.num_ans,boxes.shape[1], boxes.shape[2])
                
                outputs = model(qa_tokens, features, boxes)
                outputs = outputs.reshape(-1, dataset.num_ans)

                loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, scores, reduction="mean") * scores.size(1)
                i+=1

                score = compute_score_with_logits(outputs, scores.data).sum()

                eval_score += score.item()
                total_loss += loss.item() * features.size(0)

                final_loss = total_loss / (i * features.size(0))
                final_score = eval_score / (i * features.size(0))

    return final_loss, final_score

if __name__=="__main__":
    
    from torch.utils.data import DataLoader
    from model import VEQ
    class AttributeDict(dict):
            __getattr__ = dict.__getitem__
            __setattr__ = dict.__setitem__
            __delattr__ = dict.__delitem__

    parser = ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()

    cfg = AttributeDict(json.load(open(args.config)))

    dataset = VEQADataset("train", cfg.base_dir, cfg.train_q, cfg.train_a, cfg.features, cfg.boxes)
    eval_dataset = dataset = VEQADataset("eval", cfg.base_dir, cfg.eval_q, cfg.eval_a, cfg.features, cfg.boxes)
    # dataset = VEQADataset("train", "/home/meghana/meg/VEQA/data", "MultipleChoice_mscoco_train2014_questions.json", "mscoco_train2014_annotations.json", "coco/trainval.zarr", "coco/trainval_boxes.zarr")
    
    loader = DataLoader(dataset, cfg.batch_size, shuffle=True, num_workers=1, collate_fn=utils.trim_collate)
    eval_loader = DataLoader(eval_dataset, cfg.batch_size, shuffle=True, num_workers=1, collate_fn=utils.trim_collate)
    
    model = VEQ(cfg)
    model = model.cuda()

    optim = Adam(model.parameters(), lr=1e-5)
    optim.zero_grad()

    best_eval_score = 0
    
    for epoch in range(cfg.epochs):

        print(f"Epoch {epoch}/{cfg.epochs}:\n")
        model.train()
        train_score = 0
        total_loss = 0

        for i, (imid, qid, q_type, answer, qa_tokens_item, features, boxes, scores) in tqdm(enumerate(loader)):
            qa_tokens, qa_tokens_padded, qa_tokens_ids = qa_tokens_item["input_ids"].cuda(), qa_tokens_item["attention_mask"].cuda(), qa_tokens_item["token_type_ids"].cuda()
            # qa_tokens = qa_tokens.cuda()
            features = features.cuda()
            boxes = boxes.cuda()
            scores = scores.cuda()

            qa_tokens_item["input_ids"] = qa_tokens.reshape(qa_tokens.shape[0] * dataset.num_ans, -1)
            qa_tokens_item["attention_mask"] = qa_tokens_padded.reshape(qa_tokens_padded.shape[0] * dataset.num_ans, -1)
            qa_tokens_item["token_type_ids"] = qa_tokens_ids.reshape(qa_tokens_ids.shape[0] * dataset.num_ans, -1)
            features_r = features.repeat(1, dataset.num_ans, 1)
            features = features_r.reshape(features.shape[0] * dataset.num_ans, features.shape[1], features.shape[2])
            boxes_r = boxes.repeat(1, dataset.num_ans , 1)
            boxes = boxes_r.reshape(boxes.shape[0] * dataset.num_ans,boxes.shape[1], boxes.shape[2])
            
            outputs = model(qa_tokens_item, features, boxes)
            outputs = outputs.reshape(-1, dataset.num_ans)

            loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, scores, reduction="mean") * scores.size(1)
            i+=1

            loss.backward()
            optim.step()
            optim.zero_grad()

            score = compute_score_with_logits(outputs, scores.data).sum()
            # print("Score for this epoch: %d"%(score))
            train_score += score.item()
            total_loss += loss.item() * features.size(0)

            if i != 0 and i % 500 == 0:
                print(
                    'training: %d/%d, train_loss: %.6f, train_acc: %.3f%%' %
                    (i, len(loader), total_loss / (i * features.size(0)),
                     100 * train_score / (i * features.size(0))))

        total_loss /= len(loader)
        model.eval()
        eval_loss, eval_score = evaluate(model, eval_loader, cfg)
        model.train()

        print('eval score: %.3f%%' % (100 * eval_score))

        if (eval_score > best_eval_score):
            print("Saving best model")
            model_path = os.path.join(cfg.output_dir, "VEQA_best_model.pth")
            utils.save_model(model_path, model, epoch, optim)
            best_eval_score = eval_score