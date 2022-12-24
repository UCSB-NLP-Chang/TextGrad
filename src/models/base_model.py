import numpy as np
import torch
import torch.nn.functional as F

class BaseModel():
    def predict(self, sentence1_list, sentence2_list = None):
        '''
        sentence1_list: list of str, each element is a sentence for prediction
        sentence2_list(optional): list of str for hypothesis in NLI tasks.
        '''
        raise NotImplementedError
    
    def predict_via_embedding(self, embedding_mat, attention_mask, token_type_ids = None, labels = None):
        '''
        embedding_mat:  [batch_size, seq_len, hidden_dim], a tensor
        attention_mask: [batch_size, seq_len,]
        labels: pbatch_size,],  a tensor
        '''
        raise NotImplementedError
    
    def get_input_embedding(self,):
        raise NotImplementedError


