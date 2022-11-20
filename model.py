from transformers import LxmertTokenizer, LxmertModel
import torch.nn as nn


class VEQ(nn.Module):
    def __init__(self):
        self.lxmert = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
        self.tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
        self.classifier = nn.Linear()
        self.softmax = nn.Softmax()
        

    def forward(self, batch):
        
        pass

if __name__ == "__main__":
    lxmert = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
    tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    outputs = lxmert(**inputs)
    last_hidden_states = outputs.last_hidden_state
    print(inputs)