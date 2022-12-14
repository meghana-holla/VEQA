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
from torchmetrics.functional import precision_recall
import random

# Function for computing the top 1 and top 2 accuracy
def compute_score_with_logits(outputs, scores, k=2):
    # Top 2 accuracy compuation
    s_top_2 = (torch.argmax(scores, dim=1).repeat(1,k).reshape(-1,k)==torch.topk(outputs, k=k).indices).any(dim=1)
    
    # Top 1 accuracy compuation
    s_top_1 = torch.argmax(outputs, dim=1)==torch.argmax(scores, dim=1)
    
    return s_top_2>0, s_top_1>0


# Evaluation/Inference Submodule
def evaluate(model, loader, cfg):
    model.eval()

    eval_score = 0
    eval_score_top_2 = 0
    eval_pre = 0
    eval_rec = 0
    total_loss = 0
    total_count = 0


    for i, (imid, qid, q_type, answer, qa_tokens_item, features, boxes, scores) in tqdm(enumerate(loader)):
        qa_tokens, qa_tokens_padded, qa_tokens_ids = qa_tokens_item["input_ids"].cuda(), qa_tokens_item["attention_mask"].cuda(), qa_tokens_item["token_type_ids"].cuda()

        features = features.cuda()
        boxes = boxes.cuda()
        scores = scores.cuda()

        # Get hypothesis tokens, attention masks and token type IDs for LXMERT.
        qa_tokens_item["input_ids"] = qa_tokens.reshape(qa_tokens.shape[0] * cfg.num_ans, -1)
        qa_tokens_item["attention_mask"] = qa_tokens_padded.reshape(qa_tokens_padded.shape[0] * cfg.num_ans, -1)
        qa_tokens_item["token_type_ids"] = qa_tokens_ids.reshape(qa_tokens_ids.shape[0] * cfg.num_ans, -1)
    
        # Create image features for every qustion-answer hypothesis ie., for a batch size b, have b * cfg.num_ans datapoints, such that each datapoint is a sinlge entailment data example.
        features_r = features.repeat(1, cfg.num_ans, 1)
        features = features_r.reshape(features.shape[0] * cfg.num_ans, features.shape[1], features.shape[2])
        boxes_r = boxes.repeat(1, cfg.num_ans , 1)
        boxes = boxes_r.reshape(boxes.shape[0] * cfg.num_ans,boxes.shape[1], boxes.shape[2])
        
        # Generate visual attention mask for coco features
        visual_attention_mask = (features.sum(2)>0).int()
        
        # Get model prediction
        outputs = model(qa_tokens_item, features, boxes, visual_attention_mask)
        
        # reshape outputs from (batch * cfg.num_ans, sequence length, 1) to (batch, sequence length, cfg.num_ans)
        outputs = outputs.reshape(-1, cfg.num_ans)

        # Compute loss
        loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, scores, reduction="mean")
        
        #  Compute top 1 and top 2 accuracy
        scores_top_2, scores_top_1 = compute_score_with_logits(outputs, scores.data)
        scores_top_2, scores_top_1 = scores_top_2.sum(), scores_top_1.sum()

        # Compute precision and recall
        precision, recall = precision_recall(torch.argmax(outputs, dim=1), torch.argmax(scores, dim=1), average="macro", num_classes = cfg.num_ans)
        
        eval_pre+=precision
        eval_rec+=recall

        eval_score += scores_top_1.item()
        eval_score_top_2 += scores_top_2.item()

        total_count += outputs.size(0)
        total_loss += loss.item() 

        # Compute epoch-level average metrics values
        final_loss = total_loss / (i+1)
        final_score = eval_score / total_count
        final_score_top_2 = eval_score_top_2 / total_count
        final_pre = eval_pre / (i+1)
        final_rec = eval_rec / (i+1)

        if i == len(loader) or  i%20==19:
            print("Acc: %.3f Top-2 Acc: %.3f Rec: %.3f Prec: %.3f Loss: %.3f"%(100*final_score, 100*final_score_top_2, 100*final_rec, 100*final_pre, final_loss))

    return final_loss, final_score, final_score_top_2, final_pre, final_rec

if __name__=="__main__":
    
    import sys
    from argparse import ArgumentParser
    from torch.utils.data import DataLoader
    from model import VEQ

    parser = ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--config")
    parser.add_argument("--mode", default="test")
    args = parser.parse_args()


    # =============================Evaluation/Inference Branch =============================
    if args.mode == "test":

        class AttributeDict(dict):
                __getattr__ = dict.__getitem__
                __setattr__ = dict.__setitem__
                __delattr__ = dict.__delitem__

        cfg = AttributeDict(json.load(open(args.config)))

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)

        eval_dataset = VEQADataset("eval", cfg.base_dir, cfg.eval_q, cfg.eval_a, cfg.features, cfg.boxes, cfg)
        eval_loader = DataLoader(eval_dataset, cfg.batch_size, shuffle=True, num_workers=1, collate_fn=utils.trim_collate)
        
        model = VEQ(cfg)
        model = model.cuda()

        model_data = torch.load(args.model)
        model.load_state_dict(model_data.get('model_state', model_data))

        optim = Adam(model.parameters(), lr=1e-5)
        optim.zero_grad()
        
        model.eval()
        final_loss, final_score, final_score_top_2, final_pre, final_rec = evaluate(model, eval_loader, cfg)

        print('eval score: %.3f%% | eval top-2 score: %.3f%% | eval precision: %.3f%% | eval recall: %.3f%%' % (100*final_score, 100*final_score_top_2, 100*final_pre, 100*final_rec))


    # =============================Training Branch =============================
    elif args.mode=="train":
        from torch.utils.data import DataLoader
        
        from model import VEQ
        class AttributeDict(dict):
                __getattr__ = dict.__getitem__
                __setattr__ = dict.__setitem__
                __delattr__ = dict.__delitem__

        cfg = AttributeDict(json.load(open(args.config)))

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)

        # Create train and validation datasets
        dataset = VEQADataset("train", cfg.base_dir, cfg.train_q, cfg.train_a, cfg.features, cfg.boxes, cfg)
        eval_dataset = dataset = VEQADataset("eval", cfg.base_dir, cfg.eval_q, cfg.eval_a, cfg.features, cfg.boxes, cfg)
        
        # Load train and validation datasets
        loader = DataLoader(dataset, cfg.batch_size, shuffle=True, num_workers=1, collate_fn=utils.trim_collate)
        eval_loader = DataLoader(eval_dataset, cfg.batch_size, shuffle=True, num_workers=1, collate_fn=utils.trim_collate)
        
        model = VEQ(cfg)
        model = model.cuda()

        # Optimizer initiliazation with learning rates
        optim = Adam(model.parameters(), lr=1e-5)
        optim.zero_grad()

        best_eval_score = 0
        results = {"accuracy":[], "top 2 accuracy":[], "recall":[], "precision":[]}
        
        # Training Starts
        for epoch in range(cfg.epochs):

            print(f"=============================Epoch {epoch}/{cfg.epochs}=============================\n")
            model.train()
            train_score = 0
            train_score_top_2 = 0
            train_rec = 0
            train_pre = 0
            total_loss = 0
            total_count = 0

            for i, (imid, qid, q_type, answer, qa_tokens_item, features, boxes, scores) in tqdm(enumerate(loader)):
                qa_tokens, qa_tokens_padded, qa_tokens_ids = qa_tokens_item["input_ids"].cuda(), qa_tokens_item["attention_mask"].cuda(), qa_tokens_item["token_type_ids"].cuda()

                features = features.cuda()
                boxes = boxes.cuda()
                scores = scores.cuda()

                # Get hypothesis tokens, attention masks and token type IDs for LXMERT.
                qa_tokens_item["input_ids"] = qa_tokens.reshape(qa_tokens.shape[0] * dataset.num_ans, -1)
                qa_tokens_item["attention_mask"] = qa_tokens_padded.reshape(qa_tokens_padded.shape[0] * dataset.num_ans, -1)
                qa_tokens_item["token_type_ids"] = qa_tokens_ids.reshape(qa_tokens_ids.shape[0] * dataset.num_ans, -1)

                # Create image features for every qustion-answer hypothesis ie., for a batch size b, have b * cfg.num_ans datapoints, such that each datapoint is a sinlge entailment data example.
                features_r = features.repeat(1, dataset.num_ans, 1)
                features = features_r.reshape(features.shape[0] * dataset.num_ans, features.shape[1], features.shape[2])
                boxes_r = boxes.repeat(1, dataset.num_ans , 1)
                boxes = boxes_r.reshape(boxes.shape[0] * dataset.num_ans,boxes.shape[1], boxes.shape[2])

                # Generate visual attention mask for coco features
                visual_attention_mask = (features.sum(2)>0).int()
                
                # Get model prediction
                outputs = model(qa_tokens_item, features, boxes, visual_attention_mask)

                # reshape outputs from (batch * cfg.num_ans, sequence length, 1) to (batch, sequence length, cfg.num_ans)
                outputs = outputs.reshape(-1, dataset.num_ans)

                # Compute loss
                loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, scores, reduction="mean")
                i+=1

                loss.backward()
                optim.step()
                optim.zero_grad()


                #  Compute top 1 and top 2 accuracy
                scores_top_2, scores_top_1 = compute_score_with_logits(outputs, scores.data)
                scores_top_2, scores_top_1 = scores_top_2.sum(), scores_top_1.sum()
                
                # Compute precision and recall
                precision, recall = precision_recall(torch.argmax(outputs, dim=1), torch.argmax(scores, dim=1), average="macro", num_classes = cfg.num_ans)
                

                train_score += scores_top_1.item()
                train_score_top_2 += scores_top_2.item()

                total_loss += loss.item()
                total_count += outputs.size(0)

                train_pre += precision
                train_rec += recall

                # Diaplay performance at every 1000th run
                if i == len(loader) or i % 1000 == 999:
                    print("epoch %d || training: %d/%d, train_loss: %.6f, train accuracy: %.3f%% train top-2 accuracy: %.3f%% train precision: %.3f%% train recall: %.3f%%"%(
                        epoch, 
                        i+1, 
                        len(loader), 
                        total_loss / (i+1),
                        100 * train_score / total_count,
                        100 * train_score_top_2 / total_count,
                        100 * train_pre / (i+1),
                        100 * train_rec / (i+1)))
                    

            total_loss /= len(loader)
            print(f"=============================EVAL for Epoch {epoch}/{cfg.epochs}=============================\n")
            model.eval()
            # Validation loop
            eval_loss, eval_score, eval_score_top_2, eval_pre, eval_rec = evaluate(model, eval_loader, cfg)
            model.train()

            print('val loss: %.6f, val accuracy: %.3f%%, val top-2 accuracy: %.3f%% val precision: %.3f%% val recall: %.3f%%'%(
                eval_loss,
                100 * eval_score,
                100 * eval_score_top_2,
                100 * eval_pre,
                100 * eval_rec))

            results["accuracy"].append((train_score / total_count))
            results["top 2 accuracy"].append((train_score_top_2 / total_count))
            results["recall"].append((train_rec / len(loader)).item())
            results["precision"].append((train_pre / len(loader)).item())

            # Save best model
            if (eval_score > best_eval_score):
                print("Saving best model")
                model_path = os.path.join(cfg.output_dir, "VEQA_QA_best_model.pth")
                utils.save_model(model_path, model, epoch, optim)
                best_eval_score = eval_score

        # Save epoch-wise results
        import json
        json.dump(results, open("Trained/results_qa.json", "w"))
