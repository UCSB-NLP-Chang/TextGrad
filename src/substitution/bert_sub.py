from transformers import BertTokenizer,BertForMaskedLM
import torch
import torch.nn.functional as F
from .base_sub import BaseSubstitutor
from .sub_utils import WordNetFilter

class BertSubstitutor(BaseSubstitutor):
    '''
    implementation of our current method of generaing substitution words through BERT masked language model
    1. each prediction is based on the original word, which means we do not mask the original words
    2. we use wordnet to filter the prediction to prevent using the antoynyms
    3. we will also return the masked language model loss of the substitute words for further use
    4. We will also return the mlm loss of the original words at the same time
    '''

    def __init__(self, model_type, model_dir = './model_cache/bert_model/bert_base_uncased/masklm/', 
                mask_orig = False,
                ):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_type, cache_dir = model_dir)
        self.bert_model = BertForMaskedLM.from_pretrained(model_type,cache_dir = model_dir).to('cuda')
        print("freezing parameters... ")
        for param in self.bert_model.parameters():
            param.requires_grad = False
        self.bert_id2token = self.tokenizer.ids_to_tokens
        self.special_tokens = [self.bert_id2token[x] for x in range(1996)]  ## tokens with index from 1-1996 are regarded as stopwords (in the BERT vocabulary)
        self.stopwords += self.special_tokens
        self.stopwords += ['...']
        self.antonym_filter = WordNetFilter()
        self.mask_orig = mask_orig

    def get_neighbor_list(self, word_list, site_mask = None, k = 50, threshold_pred_score = 0, pos_list = []):
        substitute_for_sentence = []
        lm_loss_for_sentence = []
        orig_lm_loss_for_sentence = []
        for i in range(len(word_list)):
            substitute_list, lm_loss_list, orig_word_lm_loss = self.parse_sentence(word_list, i, k = k, threshold_pred_score = threshold_pred_score, site_mask = site_mask, word_pos = pos_list[i])
            substitute_for_sentence.append(substitute_list)
            lm_loss_for_sentence.append(lm_loss_list)
            orig_lm_loss_for_sentence.append(orig_word_lm_loss)
        return substitute_for_sentence, lm_loss_for_sentence, orig_lm_loss_for_sentence

    def parse_sentence(self, sub_words, position = 0, k = 50, threshold_pred_score = 0, site_mask: torch.LongTensor = None, word_pos = 'n'):
        '''
        sentence: str or word_list
        Note: 1. position here can from 0 to L+1, where L is the length of the sequence. The reason is we insert [CLS] token and [SEP] token
              to the sequence. The two special tokens are in the self.stopwords list, which will be ignored. That means we can allow position = 0/L+1

              2. We only return the probability (after softmax) rather than directly calculate the cross entropy loss. The MLM loss for each
              word will be calculated during the attack.

        '''
        if site_mask is not None:
            assert len(sub_words) == len(site_mask)
        if site_mask is not None and site_mask[position] == 0:
            return [], [], 0
        sub_words = sub_words[:]
        
        original_word = sub_words[position]
        if original_word in self.stopwords:
            return [], [], 0
        original_index = self.tokenizer.convert_tokens_to_ids(original_word)

        if self.mask_orig:
            sub_words[position] = self.tokenizer.mask_token
        assert sub_words[0] == self.tokenizer.cls_token
        assert sub_words[-1] == self.tokenizer.sep_token
        input_ids_ = torch.tensor([self.tokenizer.convert_tokens_to_ids(sub_words)])
        word_predictions = self.bert_model(input_ids_.to(self.bert_model.device))[0].squeeze()[position]  # vocab
        word_predictions_softmax = F.softmax(word_predictions, dim = -1)
        word_pred_scores_all, word_predictions = torch.topk(word_predictions, k, -1)  # k
        top_k_word_predictions_softmax, _ = torch.topk(word_predictions_softmax, k, -1)  # k

        filtered_list = []
        filtered_probs = []
        filtered_detokenized_words = []

        substitutes = word_predictions
        word_pred_scores = word_pred_scores_all
        substitutes, probability_list, detokenized_words = self.get_substitues(substitutes, word_pred_scores, top_k_word_predictions_softmax, threshold_pred_score)
        for i in range(len(substitutes)):
            substitute = substitutes[i]
            if substitute == original_word:
                continue  # filter out original word
            if '##' in substitute:
                continue  # filter out sub-word
            if substitute in self.stopwords:
                continue    
            filtered_list.append(substitute)
            filtered_probs.append(probability_list[i])
            filtered_detokenized_words.append(detokenized_words[i])
        
        antonym_filtered_list = []
        antonym_filtered_probs = []
        detokenized_orig_word = self.tokenizer.decode([original_index])
        antonym_filtered_index_list = self.antonym_filter.filter_antonym(detokenized_orig_word, filtered_detokenized_words, word_pos)
        for idx in antonym_filtered_index_list:
            antonym_filtered_list.append(filtered_list[idx])
            antonym_filtered_probs.append(filtered_probs[idx])
        return antonym_filtered_list, antonym_filtered_probs, word_predictions_softmax[original_index]


    def get_substitues(self, substitutes, substitute_score = None, substitute_probability = None, threshold = 3.0):
        words = []
        probability_list = []
        detokenized_words = []
        for (i,j, k) in zip(substitutes, substitute_score, substitute_probability):
            if threshold != 0 and j < threshold:
                break
            words.append(self.tokenizer._convert_id_to_token(int(i)))
            detokenized_words.append(self.tokenizer.decode([int(i)]))
            probability_list.append(k.item())

        return words, probability_list, detokenized_words


