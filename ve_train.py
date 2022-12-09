import json
import os
import zarr
from tqdm import tqdm
import torch
import numpy as np
from torch.nn import BCELoss

from torch.utils.data import Dataset
from torch.optim import Adam
from transformers import LxmertTokenizer, LxmertModel
import utils

from argparse import ArgumentParser

from ve_dataset import VEQADataset


def compute_score_with_logits(outputs, scores):
    # s = outputs*scores
    return (outputs>0.5).int()==scores.int()
    # s = torch.argmax(outputs, dim=1)==torch.argmax(scores, dim=1)
    return s>0

def evaluate(model, loader, cfg):
    model.eval()

    eval_score = 0
    total_loss = 0
    total_size = 0
    loss_fn = BCELoss(reduction="sum")
    
    for i, (features, boxes, qa_tokens_item, scores) in tqdm(enumerate(loader)):
        qa_tokens, qa_tokens_padded, qa_tokens_ids = qa_tokens_item["input_ids"].cuda(), qa_tokens_item["attention_mask"].cuda(), qa_tokens_item["token_type_ids"].cuda()
        # qa_tokens = qa_tokens.cuda()
        features = features.cuda()
        boxes = boxes.cuda()
        scores = scores.cuda()

        qa_tokens_item["input_ids"] = qa_tokens.squeeze(1) if len(qa_tokens.shape)>2 else qa_tokens
        qa_tokens_item["attention_mask"] = qa_tokens_padded.squeeze(1) if len(qa_tokens_padded.shape)>2 else qa_tokens_padded
        qa_tokens_item["token_type_ids"] = qa_tokens_ids.squeeze(1) if len(qa_tokens_ids.shape)>2 else qa_tokens_ids

        visual_attention_mask = (features.sum(2)>0).int()

        # qa_tokens_item["input_ids"] = qa_tokens.reshape(qa_tokens.shape[0] * dataset.num_ans, -1)
        # qa_tokens_item["attention_mask"] = qa_tokens_padded.reshape(qa_tokens_padded.shape[0] * dataset.num_ans, -1)
        # qa_tokens_item["token_type_ids"] = qa_tokens_ids.reshape(qa_tokens_ids.shape[0] * dataset.num_ans, -1)
    

        # qa_tokens = qa_tokens.reshape(qa_tokens.shape[0] * dataset.num_ans, -1)
        # features_r = features.repeat(1, dataset.num_ans, 1)
        # features = features_r.reshape(features.shape[0] * dataset.num_ans, features.shape[1], features.shape[2])
        # boxes_r = boxes.repeat(1, dataset.num_ans , 1)
        # boxes = boxes_r.reshape(boxes.shape[0] * dataset.num_ans,boxes.shape[1], boxes.shape[2])
        # print(features.shape, boxes.shape, visual_attention_mask.shape, qa_tokens["input_ids"].shape, qa_tokens["attention_mask"].shape, qa_tokens["token_type_ids"].shape)
        outputs = model(qa_tokens_item, features, boxes, visual_attention_mask).squeeze(1)
        # outputs = outputs.reshape(-1, dataset.num_ans)

        loss = loss_fn(outputs, scores.float()) #* scores.size(1)
        i+=1

        score = compute_score_with_logits(outputs, scores.data).sum()
        total_size += features.size(0)

        eval_score += score.item()
        total_loss += loss.item()

        final_loss = total_loss / total_size
        final_score = eval_score / total_size

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
    results = {"train":{"loss":[], "accuracy":[]}, "eval":{"loss":[], "accuracy":[]}}

    dataset = VEQADataset("train")
    eval_dataset = VEQADataset("eval")

    loader = DataLoader(dataset, cfg.batch_size, shuffle=True, num_workers=1, collate_fn=utils.trim_collate)
    eval_loader = DataLoader(eval_dataset, cfg.batch_size, shuffle=True, num_workers=1, collate_fn=utils.trim_collate)
    
    model = VEQ(cfg)
    model = model.cuda()

    optim = Adam(model.parameters(), lr=1e-5)
    optim.zero_grad()

    loss_fn = BCELoss(reduction="sum")

    best_eval_score = 0
    eval_loss, eval_score = 0, 0

    for epoch in range(cfg.epochs):

        print(f"Epoch {epoch}/{cfg.epochs}:\n")
        model.train()
        train_score = 0
        total_loss = 0
        total_size = 0
        
        

        for i, (features, boxes, qa_tokens_item, scores) in tqdm(enumerate(loader)):
            qa_tokens, qa_tokens_padded, qa_tokens_ids = qa_tokens_item["input_ids"].cuda(), qa_tokens_item["attention_mask"].cuda(), qa_tokens_item["token_type_ids"].cuda()
            
            features = features.cuda()
            boxes = boxes.cuda()
            scores = scores.cuda()

            qa_tokens_item["input_ids"] = qa_tokens.squeeze(1) if len(qa_tokens.shape)>2 else qa_tokens
            qa_tokens_item["attention_mask"] = qa_tokens_padded.squeeze(1) if len(qa_tokens_padded.shape)>2 else qa_tokens_padded
            qa_tokens_item["token_type_ids"] = qa_tokens_ids.squeeze(1) if len(qa_tokens_ids.shape)>2 else qa_tokens_ids

            visual_attention_mask = (features.sum(2)>0).int()

            outputs = model(qa_tokens_item, features, boxes, visual_attention_mask).squeeze()

            loss = loss_fn(outputs, scores.float())
            i+=1

            loss.backward()
            optim.step()
            optim.zero_grad()

            

            score = compute_score_with_logits(outputs, scores.data).sum()

            train_score += score.item()
            total_loss += loss.item()
            total_size += features.size(0)

            if i != 0 and i % 1000 == 0:
                print(
                    'training: %d/%d, train_loss: %.3f, train_acc: %.3f%%' %
                    (i, len(loader), total_loss / total_size,
                     100 * train_score / total_size))

        total_loss /= total_size
        model.eval()
        eval_loss, eval_score = evaluate(model, eval_loader, cfg)
        model.train()

        print('eval loss: %.3f eval score: %.3f%%' % (eval_loss, 100 * eval_score))

        if (eval_score > best_eval_score):
            print("Saving best model")
            model_path = os.path.join(cfg.output_dir, "VE_best_model.pth")
            utils.save_model(model_path, model, epoch, optim)
            best_eval_score = eval_score

        results["eval"]["loss"].append(eval_loss)
        results["eval"]["accuracy"].append(eval_score)
        results["train"]["loss"].append(total_loss)
        results["train"]["accuracy"].append(train_score/total_size)
    
        import json
        with open("/home/meghana/meg/VEQA/Trained/ve_train_results.json", "w") as f:
            json.dump(results, f)