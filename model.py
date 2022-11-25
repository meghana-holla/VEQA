from transformers import LxmertTokenizer, LxmertModel
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

class VEQ(nn.Module):
    def __init__(self, cfg):
        super(VEQ, self).__init__()
        self.lxmert = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
        self.tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
        self.classifier = nn.Linear(768, 1)
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()
        self.norm = weight_norm
        self.sigmoid = nn.Sigmoid()
        self.num_ans = cfg.num_ans

        

    def forward(self, qa_tokens, features, boxes):

        # qa_tokens = qa_tokens.reshape(qa_tokens.shape[0] * self.num_ans, -1)
        # features_r = features.repeat(1, self.num_ans, 1)
        # features = features_r.reshape(features.shape[0] * self.num_ans, features.shape[1], features.shape[2])
        # boxes_r = boxes.repeat(1, self.num_ans , 1)
        # boxes = boxes_r.reshape(boxes.shape[0] * self.num_ans,boxes.shape[1], boxes.shape[2])

        output = self.lxmert(visual_feats=features, visual_pos=boxes, input_ids=qa_tokens["input_ids"], attention_mask=qa_tokens["attention_mask"], token_type_ids=qa_tokens["token_type_ids"])
        output_next = output.pooled_output
        output = self.sigmoid(self.classifier(output_next))
        # output = self.norm(output, dim=None)
        
        return output

if __name__ == "__main__":
    lxmert = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
    tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    outputs = lxmert(**inputs)
    last_hidden_states = outputs.last_hidden_state
    print(inputs)