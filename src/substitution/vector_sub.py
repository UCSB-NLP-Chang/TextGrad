from .base_sub import BaseSubstitutor
import pickle

class VectorSub(BaseSubstitutor):
    def __init__(self, file_path = './pkl'):
        '''
        file_path: a pickel file which contains substitution for each words as a dict
        {
            key: string, a word
            values: list of string, substitute words for the key
        }
        '''
        super().__init__()
        with open(file_path,'rb') as f:
            self.sub_dict = pickle.load(f)
    def get_subword(self, word, pos = None):
        if word in self.stopwords:
            return [], []
        if word in self.sub_dict:
            neighbor_list = self.sub_dict[word]
        else:
            neighbor_list = []
        lm_loss = [0 for _ in neighbor_list]
        return neighbor_list, lm_loss
    
    def get_neighbor_list(self, word_list, *args):
        return super().get_neighbor_list(word_list, *args)
