import numpy as np
import torch
import torch.nn.functional as F
from ..substitution.base_sub import BaseSubstitutor
from transformers import PreTrainedTokenizer
from ..models.base_model import BaseModel
from .attack_util import (match_subword_with_word, match_subword_with_word_albert, 
                            pos_tag, expand_with_match_index, match_subword_with_word_roberta,
                          STESample, STERandSelect, adjust_discrete_num)
import time
from typing import List

class PGDAttack():
    def __init__(self, victim_model: BaseModel, tokenizer: PreTrainedTokenizer, substitutor:BaseSubstitutor,
                    eta_z = 0.8, eta_u = 0.8, modification_rate = 0.25, iter_time = 20, max_neighbor_num = 50,
                    final_sample_time = 20, ste = True, norm = True, rand_init = True, no_subword = False, 
                    multi_sample = False, discrete_sample_num = 20, use_lm = False, lm_loss_beta = 0.1, use_cw_loss = True, 
                    device = torch.device("cuda"), num_classes = 2, victim = 'bert', use_cache = False, sentence_pair = False
                    ):
        self.victim_model = victim_model
        self.tokenizer = tokenizer
        self.substitutor = substitutor
        self.eta_z = eta_z
        self.eta_u = eta_u
        self.modification_rate = modification_rate

        self.iter_time = iter_time
        self.max_neighbor_num = max_neighbor_num
        self.final_sample_time = final_sample_time
        self.ste = ste
        self.norm = norm
        self.rand_init = rand_init
        self.no_subword = no_subword
        self.use_lm = use_lm

        self.multi_sample = multi_sample
        self.discrete_sample_num = discrete_sample_num

        self.lm_loss_beta = lm_loss_beta
        self.use_cw_loss = use_cw_loss
        self.use_cache = use_cache

        self.cw_tau = 0

        self.patience = 0
        self.device = device
        self.num_classes = num_classes
        self.victim = victim
        self.sentence_pair = sentence_pair
        self.input_embedding = self.victim_model.get_input_embedding()
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction = 'none')

    def tokenize_sentence(self, sentence1, sentence2 = None):
        '''
        sentence1: str
        sentence2: str
        '''
        result = self.tokenizer(text = sentence1, text_pair = sentence2, add_special_tokens = True, truncation = True)
        idx_list = result['input_ids']
        attention_mask = result['attention_mask']
        token_type_ids = result['token_type_ids'] if 'token_type_ids' in result else None
        token_list = self.tokenizer.convert_ids_to_tokens(idx_list)
        assert len(token_list) == len(idx_list)
        output = (token_list, idx_list, attention_mask, token_type_ids)
        if sentence2 != None:
            sep_loc = token_list.index(self.tokenizer.sep_token)
            sentence1_tokens = token_list[1:sep_loc]
            sentence2_tokens = token_list[sep_loc + 1: -1]
            output += (sentence1_tokens, sentence2_tokens)
        else:
            sentence1_tokens = token_list[1:-1]
            output += (sentence1_tokens, )
        return output

    def detokenize_idxs(self, idx_list):
        sentence = self.tokenizer.decode(idx_list, skip_special_tokens = True)
        return sentence
    
    def get_subtoken_mask(self, subword_list: List[str]):
        orig_mask = [1 for _ in subword_list]
        for idx in range(len(subword_list)):
            if subword_list[idx].startswith("##"):
                orig_mask[idx] = 0
                if idx > 0 and orig_mask[idx - 1] == 1:
                    orig_mask[idx - 1] = 0
        return orig_mask

    def build_neighbor_matrix(self, subword_list, match_index, pos_list):
        '''
        Build the substitution matrix for variable $z$ and $u$. The variable `mat` is a L*N matrix where L is the sequence length and N is the neighbor number.
        `mat` contains the substitution tokens for each token in the original sentence
        `site_mask` (a vector with L dimension) indicates which parts of tokens should be skipped during attacking, such as stopwords
        `sub_mask` is a L*N matrix indicating which part of substitution tokens should be skipped. For example, if a substituion token is stopwords, sub-words, etc, we 
        will not use it to replace the original token for attacking
        '''
        mat = torch.zeros([len(subword_list), self.max_neighbor_num], device = self.device, dtype = torch.long)
        subword_score_mat = torch.ones([len(subword_list), self.max_neighbor_num], device = self.device, dtype = torch.float32)
        origword_score_mat = torch.ones([len(subword_list), 1], device = self.device, dtype = torch.float32)
        site_mask = torch.ones([len(subword_list)], device = self.device, dtype = torch.long)

        if match_index is not None:
            ## As we have explained before, we match the POS of original words with its sub-words.
            expanded_pos_list = expand_with_match_index(pos_list, match_index)
            pos_mask = [1 if x in ['r','a','n','v'] else 0 for x in pos_list]
            expanded_pos_mask = expand_with_match_index(pos_mask, match_index)
            expanded_pos_mask = torch.tensor(expanded_pos_mask, device = self.device)
            site_mask *= expanded_pos_mask    ## mask those stopwords
        else:
            expanded_pos_list = ['none' for _ in range(len(subword_list))]

        ## the substitutor module will generate the substitution tokens as well as their MLM loss for regularization.
        substitute_for_sentence, lm_loss_for_sentence, orig_lm_loss_for_sentence = self.substitutor.get_neighbor_list(subword_list, site_mask, pos_list = expanded_pos_list)

        for i in range(mat.size(0)):
            if site_mask[i] == 0:
                continue
            substitute_words = substitute_for_sentence[i]
            lm_loss_for_word = lm_loss_for_sentence[i]
            orig_lm_loss = orig_lm_loss_for_sentence[i]
            if len(substitute_words) > self.max_neighbor_num:
                substitute_words = substitute_words[:self.max_neighbor_num]
                lm_loss_for_word = lm_loss_for_word[:self.max_neighbor_num]
            for j in range(len(substitute_words)):
                curr_subword = substitute_words[j]
                curr_subtokens = self.tokenizer.tokenize(curr_subword)
                substitute_token_id = self.tokenizer.convert_tokens_to_ids(curr_subtokens)
                if len(substitute_token_id) >= 2:
                    continue
                mat[i][j] = substitute_token_id[0]
                subword_score_mat[i][j] = lm_loss_for_word[j]
                origword_score_mat[i][0] = orig_lm_loss
        sub_mask = mat != 0

        zero_substitute_pos_mask = torch.sign(torch.sum(sub_mask, dim = 1))
        site_mask *= zero_substitute_pos_mask

        return mat, site_mask, sub_mask, subword_score_mat, origword_score_mat

    def build_sentence_pair_neighbor_matrix(self, subword_list, match_index, pos_list, sentence1_tokens, sentence2_tokens):
        '''
        see `build_neighbor_matrix` for explanation. The only difference is the first sentence will be all masked.
        '''
        mat = torch.zeros([len(subword_list), self.max_neighbor_num], device = self.device, dtype = torch.long)
        subword_score_mat = torch.ones([len(subword_list), self.max_neighbor_num], device = self.device, dtype = torch.float32)
        origword_score_mat = torch.ones([len(subword_list), 1], device = self.device, dtype = torch.float32)
        site_mask = torch.ones([len(subword_list)], device = self.device, dtype = torch.long)

        if match_index is not None:
            expanded_pos_list = expand_with_match_index(pos_list, match_index)
            pos_mask = [1 if x in ['r','a','n','v'] else 0 for x in pos_list]
            expanded_pos_mask = expand_with_match_index(pos_mask, match_index)
            expanded_pos_mask = torch.tensor(expanded_pos_mask, device = self.device)
            site_mask *= expanded_pos_mask
        else:
            expanded_pos_list = ['none' for _ in range(len(subword_list))]

        only_sen2_tokens = [self.tokenizer.cls_token] + sentence2_tokens + [self.tokenizer.sep_token]
        only_sen2_sitemask = site_mask[1 + len(sentence1_tokens):]
        only_sen2_poslist = expanded_pos_list[1 + len(sentence1_tokens):]
        assert len(only_sen2_tokens) == len(only_sen2_sitemask)
        substitute_for_sentence, lm_loss_for_sentence, orig_lm_loss_for_sentence = self.substitutor.get_neighbor_list(only_sen2_tokens, only_sen2_sitemask, pos_list = only_sen2_poslist)
        assert len(substitute_for_sentence) == len(only_sen2_tokens)

        for i in range(1 + len(sentence1_tokens), mat.size(0)):
            if site_mask[i] == 0:
                continue
            substitute_words = substitute_for_sentence[i - 1 - len(sentence1_tokens)]
            lm_loss_for_word = lm_loss_for_sentence[i - 1 - len(sentence1_tokens)]
            orig_lm_loss = orig_lm_loss_for_sentence[i - 1 - len(sentence1_tokens)]
            if len(substitute_words) > self.max_neighbor_num:
                substitute_words = substitute_words[:self.max_neighbor_num]
                lm_loss_for_word = lm_loss_for_word[:self.max_neighbor_num]
            for j in range(len(substitute_words)):
                curr_subword = substitute_words[j]
                curr_subtokens = self.tokenizer.tokenize(curr_subword)
                substitute_token_id = self.tokenizer.convert_tokens_to_ids(curr_subtokens)
                if len(substitute_token_id) >= 2 or len(substitute_token_id) <= 0:
                    continue
                mat[i][j] = substitute_token_id[0]
                subword_score_mat[i][j] = lm_loss_for_word[j]
                origword_score_mat[i][0] = orig_lm_loss
        sub_mask = mat != 0

        zero_substitute_pos_mask = torch.sign(torch.sum(sub_mask, dim = 1))
        site_mask *= zero_substitute_pos_mask

        return mat, site_mask, sub_mask, subword_score_mat, origword_score_mat

    def get_word_list(self, sentence):
        if type(sentence) == str:
            word_list = sentence.split()
        else:
            word_list = sentence
        return word_list
    
    def init_perturb_tensor(self, site_mask, sub_mask, init_per_sample = 1, project = True):
        '''
        initialize the attack variable $z$ and $u$
        '''
        seq_len, neighbor_num = sub_mask.size()
        z_tensor = torch.ones([init_per_sample, seq_len], device = self.device, requires_grad = True, dtype = torch.double)
        u_tensor = torch.zeros([init_per_sample, seq_len, neighbor_num], device = self.device, dtype = torch.double).fill_(1/neighbor_num)

        if self.rand_init:
            torch.nn.init.uniform_(z_tensor, 0, 1)
            torch.nn.init.uniform_(u_tensor, 0, 1)

        site_mask_multi_init = site_mask.view(1, seq_len).repeat(init_per_sample, 1)
        sub_mask_multi_init = sub_mask.view(1, seq_len, neighbor_num).repeat(init_per_sample, 1, 1)
        z_tensor = (z_tensor * site_mask_multi_init).detach()
        u_tensor = (u_tensor * sub_mask_multi_init).detach()
        z_tensor = z_tensor.view(-1, seq_len)
        u_tensor = u_tensor.view(-1, seq_len, neighbor_num)
        if project:
            z_tensor = self.project_z_tensor(z_tensor, eps = self.eps).detach().clone().view(init_per_sample, seq_len)
            for i in range(init_per_sample):
                u_tensor[i] = self.project_u_tensor(u_tensor[i], site_mask = site_mask_multi_init[i], sub_mask = sub_mask_multi_init[i]).detach().clone().view(init_per_sample, seq_len, neighbor_num)

        z_tensor.requires_grad = True
        u_tensor.requires_grad = True

        return z_tensor, u_tensor
    
    def apply_perturb(self, z_tensor, u_tensor, site_mask, sub_mask, orig_embeddings, subword_embeddings, loss_incremental):
        '''
        Using current $z$ and $u$ to perturb the sentence. Will return the perturbed sentence in the embedding space.
        z_tensor / site_mask: (init_num, seq_len)
        u_tensor / sub_mask: (init_num, seq_len, neighbor_num)

        mask:      0:mask, will not be replaced  |   1:not mask, can be replaced

        orig_embeddings: (seq_len, hidden_dim)
        subword_embeddings: (seq_len, neighbor_num, hidden_dim)
        '''
        init_num, seq_len, neighbor_num = u_tensor.size()
        new_site_mask = site_mask.view(1, seq_len).repeat(init_num, 1)
        new_sub_mask = sub_mask.view(1, seq_len, neighbor_num).repeat(init_num, 1, 1)

        masked_z_tensor = z_tensor * new_site_mask
        masked_u_tensor = u_tensor * new_sub_mask
        
        if self.use_lm:
            loss_incremental = loss_incremental.view(1, seq_len, neighbor_num)
            subword_lm_loss = masked_z_tensor * torch.sum(masked_u_tensor * loss_incremental, dim = 2)     
        else:
            subword_lm_loss = None

        rand_thres = torch.rand(masked_z_tensor.size()).to(self.device)
        if self.ste:
            discrete_z = STESample.apply(masked_z_tensor - rand_thres)        
        else:
            discrete_z = masked_z_tensor        
        discrete_z = discrete_z.view(init_num, seq_len, 1)

        masked_u_tensor = masked_u_tensor.view(-1, neighbor_num)
        if self.ste:
            flat_site_mask = new_site_mask.view(-1).eq(1)
            masked_u_tensor = masked_u_tensor.view(-1, neighbor_num)
            masked_u_tensor[flat_site_mask] = STERandSelect.apply(masked_u_tensor[flat_site_mask])
            discrete_u = masked_u_tensor.view(init_num, seq_len, neighbor_num)
        else:
            discrete_u = masked_u_tensor
        discrete_u = discrete_u.view(init_num, seq_len, neighbor_num, 1)

        orig_embeddings = orig_embeddings.view(1, seq_len, -1)
        subword_embeddings = subword_embeddings.view(1, seq_len, neighbor_num, -1)
        new_embeddings = (1 - discrete_z) * orig_embeddings + discrete_z * torch.sum(discrete_u * subword_embeddings, dim = 2)

        discrete_u = discrete_u.view(init_num, seq_len, neighbor_num)
        discrete_z = discrete_z.view(init_num, seq_len)

        return new_embeddings, discrete_z, discrete_u, subword_lm_loss

    def bisection(self, a, eps, xi = 1e-5, ub=1):
        '''
        bisection method to find the root for the projection operation of $z$
        '''
        pa = torch.clip(a, 0, ub)
        if torch.sum(pa).item() <= eps:
            upper_S_update = pa
        else:
            mu_l = torch.min(a-1)
            mu_u = torch.max(a)
            while torch.abs(mu_u - mu_l)>xi:

                mu_a = (mu_u + mu_l)/2
                gu = torch.sum(torch.clip(a-mu_a, 0, ub)) - eps
                gu_l = torch.sum(torch.clip(a-mu_l, 0, ub)) - eps
                if gu == 0: 
                    break
                if torch.sign(gu) == torch.sign(gu_l):
                    mu_l = mu_a
                else:
                    mu_u = mu_a
            upper_S_update = torch.clip(a-mu_a, 0, ub)
            
        return upper_S_update

    def bisection_u(self, a, eps, xi = 1e-5, ub=1):
        '''
        bisection method to find the root for the projection operation of $u$
        '''
        pa = torch.clip(a, 0, ub)
        if np.abs(torch.sum(pa).item() - eps) <= xi:
            upper_S_update = pa
        else:
            mu_l = torch.min(a-1).item()
            mu_u = torch.max(a).item()
            while np.abs(mu_u - mu_l)>xi:
                mu_a = (mu_u + mu_l)/2
                gu = torch.sum(torch.clip(a-mu_a, 0, ub)) - eps
                gu_l = torch.sum(torch.clip(a-mu_l, 0, ub)) - eps + 1e-8
                gu_u = torch.sum(torch.clip(a-mu_u, 0, ub)) - eps
                if gu == 0: 
                    break
                elif gu_l == 0:
                    mu_a = mu_l
                    break
                elif gu_u == 0:
                    mu_a = mu_u
                    break
                if gu * gu_l < 0:  
                    mu_l = mu_l
                    mu_u = mu_a
                elif gu * gu_u < 0:  
                    mu_u = mu_u
                    mu_l = mu_a
                else:
                    print(a)
                    print(gu, gu_l, gu_u)
                    raise Exception()

            upper_S_update = torch.clip(a-mu_a, 0, ub)
            
        return upper_S_update    

    def project_z_tensor(self, z_tensor,eps):
        for i in range(z_tensor.size(0)):
            z_tensor[i] = self.bisection(z_tensor[i], eps)
            assert torch.sum(z_tensor[i]) <= eps + 1e-3,  f"{torch.sum(z_tensor[i]).item()}, {self.eps}"
        return z_tensor

    def project_u_tensor(self, u_tensor, site_mask, sub_mask):
        skip = site_mask == 0
        subword_opt = sub_mask != 0
        for i in range(u_tensor.size(0)):
            if skip[i]:
                continue
            u_tensor[i][subword_opt[i]] = self.bisection_u(u_tensor[i][subword_opt[i]], eps = 1)
            assert torch.abs(torch.sum(u_tensor[i][subword_opt[i]]) - 1) <= 1e-3
        return u_tensor

    def norm_vector(self, vec):
        if torch.sum(vec) == 0:
            return vec
        norm_vec = vec / torch.norm(vec)
        return norm_vec

    def joint_optimize(self,z_tensor, u_tensor, z_grad, u_grad, site_mask, sub_mask, iter_time):
        '''
        jointly optimize the two attack variables. The learning rate will decay with the attack iteration increases.
        '''
        z_update = self.eta_z / np.sqrt(iter_time) * z_grad
        u_update = self.eta_u / np.sqrt(iter_time) * u_grad
        z_tensor_update = z_tensor + z_update
        u_tensor_update = u_tensor + u_update
        z_tensor_list = []
        u_tensor_list = []
        for i in range(z_tensor_update.size(0)):
            z_tensor_res = self.bisection(z_tensor_update[i], eps = self.eps,)
            assert torch.sum(z_tensor_res) < self.eps + 1e-3, f"{torch.sum(z_tensor_res).item()}, {self.eps}"
            z_tensor_list.append(z_tensor_res)
        for i in range(u_tensor_update.size(0)):
            u_tensor_res = self.project_u_tensor(u_tensor_update[i], site_mask, sub_mask)
            u_tensor_list.append(u_tensor_res)

        z_tensor_res = torch.stack(z_tensor_list, dim = 0)
        u_tensor_res = torch.stack(u_tensor_list, dim = 0)
        return z_tensor_res, u_tensor_res

    def discretize_z(self, z_tensor, site_mask = None):
        z_tensor[site_mask != 1] = -10000
        rand_thres = torch.tensor(np.random.uniform(size=z_tensor.size())).to(self.device)
        discrete_z = torch.where(z_tensor > rand_thres, 1, 0)
        return discrete_z

    def discretize_u(self, u_tensor, site_mask = None):
        res = []    
        for i in range(u_tensor.size(0)):
            if site_mask[i] == 0:
                res.append(-1)
                continue
            prob = u_tensor[i].cpu().detach().numpy()
            prob = prob / np.sum(prob)
            substitute_idx = np.random.choice(u_tensor.size(1), p = prob)
            res.append(substitute_idx)
        return torch.tensor(res, device = self.device)

    def apply_substitution(self, discrete_z, discrete_u, idx_list, subword_idx_mat):
        discrete_z = np.reshape(discrete_z, newshape = [-1])
        discrete_u = np.reshape(discrete_u, newshape = [-1])
        replace_position = np.where(discrete_z == 1)[0]
        substitute_idx = discrete_u
        new_word_idx_list = idx_list[:]
        if np.sum(discrete_z) != 0:
            for i in range(len(replace_position)):
                curr_pos = replace_position[i]
                curr_subword_idx = subword_idx_mat[curr_pos][substitute_idx[curr_pos]]
                new_word_idx_list[curr_pos] = curr_subword_idx
        
        if not self.sentence_pair:
            sentence = self.detokenize_idxs(new_word_idx_list[1:-1])
        else:
            sep_index = -1
            for i in range(len(new_word_idx_list)):
                if new_word_idx_list[i] == self.tokenizer.sep_token_id:
                    sep_index = i
                    break
            assert sep_index > 0
            sentence = self.detokenize_idxs(new_word_idx_list[sep_index + 1:-1])  ## remove [CLS]/[SEP]
        return sentence

    def perturb(self, sentence, idx_list, orig_label, site_mask, sub_mask, subword_idx_mat, loss_incremental, 
                      attention_mask, token_type_ids, attack_word_num):
        if self.sentence_pair:
            sentence1, sentence2 = sentence
        seq_len, neighbor_num = subword_idx_mat.size()
        local_discrete_num = adjust_discrete_num(self.victim, seq_len, self.discrete_sample_num)

        idx_tensor = torch.LongTensor(idx_list).to(self.device)
        attention_mask = torch.tensor(attention_mask).to(self.device)
        attention_mask = torch.unsqueeze(attention_mask, 0).float()
        if token_type_ids is not None:
            token_type_ids = torch.tensor(token_type_ids).to(self.device)
            token_type_ids = torch.unsqueeze(token_type_ids, 0)

        orig_embeddings = self.input_embedding(idx_tensor)
        subword_embeddings = self.input_embedding(subword_idx_mat)

        z_tensor, u_tensor = self.init_perturb_tensor(site_mask, sub_mask, project = True, init_per_sample = 1)
        labels = torch.LongTensor([orig_label]).to(self.device)        
        succ_discrete_z_list = []
        succ_discrete_u_list = []

        for i in range(self.iter_time + 1):
            if not self.multi_sample:
                expanded_labels = labels
                new_embeddings, discrete_z, discrete_u, subword_lm_loss = self.apply_perturb(z_tensor, u_tensor, site_mask, sub_mask, orig_embeddings, subword_embeddings, loss_incremental)
                new_embeddings = new_embeddings.float()
                result = self.victim_model.predict_via_embedding(new_embeddings, attention_mask, token_type_ids, expanded_labels)
                logits = result.logits
                if not self.use_cw_loss:
                    if self.use_lm:
                        loss = result.loss + self.lm_loss_beta * subword_lm_loss.mean()
                    else:
                        loss = result.loss
                else:
                    if self.use_lm:
                        logit_mask = F.one_hot(labels, num_classes = self.num_classes)
                        logit_orig = torch.sum(logits * logit_mask,)
                        logit_others = torch.max(logits - 99999 * logit_mask)
                        cw_loss = F.relu(logit_orig - logit_others + self.cw_tau)
                        loss = cw_loss - self.lm_loss_beta * subword_lm_loss.mean()
                        loss = -loss
                    else:
                        logit_mask = F.one_hot(labels, num_classes = self.num_classes)
                        logit_orig = torch.sum(logits * logit_mask,)
                        logit_others = torch.max(logits - 99999 * logit_mask)
                        cw_loss = F.relu(logit_orig - logit_others + self.cw_tau)
                        loss = -cw_loss
                score = F.softmax(logits, dim = -1)[0]
                loss.backward(retain_graph = True)
                z_grad = z_tensor.grad
                u_grad = u_tensor.grad
                curr_model_prediction = score
            else:
                expanded_labels = labels.repeat(local_discrete_num)
                expanded_attention_mask = attention_mask.repeat(local_discrete_num, 1)
                if token_type_ids is not None:
                    expanded_token_type_ids = token_type_ids.repeat(local_discrete_num, 1)
                else:
                    expanded_token_type_ids = None
                new_embeddings_list = []
                discrete_z_list = []
                discrete_u_list = []
                subword_lm_loss_list = []
                for _ in range(local_discrete_num):
                    new_embeddings, discrete_z, discrete_u, subword_lm_loss = self.apply_perturb(z_tensor, u_tensor, site_mask, sub_mask, orig_embeddings, subword_embeddings, loss_incremental)
                    discrete_z_list.append(discrete_z)
                    discrete_u_list.append(discrete_u)
                    new_embeddings_list.append(new_embeddings)      
                    subword_lm_loss_list.append(subword_lm_loss) 
                if self.use_lm:
                    subword_lm_loss_list = torch.stack(subword_lm_loss_list, dim = 0)
                
                new_embeddings = torch.stack(new_embeddings_list, dim = 0)  ## sample_size, init_num, seq_len, hidden_dim
                new_embeddings = new_embeddings.transpose(1, 0) ## init_num, sample_size, seq_len, hidden_dim
                new_embeddings = new_embeddings.view(local_discrete_num, seq_len, -1).float()
                result = self.victim_model.predict_via_embedding(new_embeddings, expanded_attention_mask,expanded_token_type_ids)


                logits = result.logits   ##  sample_size, num_classes
                scores = F.softmax(logits, dim = -1)
                if not self.use_cw_loss:
                    loss_values = self.loss_fct(logits, expanded_labels)
                    loss_values = loss_values.view(local_discrete_num)
                else:
                    logit_mask = F.one_hot(expanded_labels, num_classes = self.num_classes)
                    logit_orig = torch.sum(logits * logit_mask, axis = -1)
                    logit_others, _ = torch.max(logits - 99999 * logit_mask, axis = -1)
                    cw_loss = F.relu(logit_orig - logit_others + self.cw_tau)
                    loss = cw_loss
                    loss_values = -cw_loss
                target_index = torch.argmax(loss_values)
                worst_score = scores[target_index]
                discrete_z = discrete_z_list[target_index]
                discrete_u = discrete_u_list[target_index]

                if self.use_lm:
                    loss = torch.mean(loss_values) + self.lm_loss_beta * torch.mean(subword_lm_loss_list)
                else:
                    loss = torch.mean(loss_values)
                loss.backward(retain_graph = True)
                
                z_grad = z_tensor.grad
                u_grad = u_tensor.grad

                curr_model_prediction = worst_score                          

            z_grad = self.norm_vector(z_grad)
            z_grad = z_grad
            for idx in range(len(site_mask)):
                if site_mask[idx] == 1:
                    u_grad[0][idx] = self.norm_vector(u_grad[0][idx])

            if torch.argmax(curr_model_prediction) != orig_label and self.ste:
                curr_discrete_z = discrete_z.detach().clone().view(-1).cpu().numpy()
                curr_discrete_u = torch.argmax(discrete_u.detach().clone(), dim = -1).view(-1).cpu().numpy()
                succ_discrete_z_list.append(curr_discrete_z)
                succ_discrete_u_list.append(curr_discrete_u)

            if i == self.iter_time:
                break

            z_tensor_opt, u_tensor_opt = self.joint_optimize(z_tensor.detach().clone(), u_tensor.detach().clone(), z_grad, u_grad, site_mask, sub_mask, i + 1)
            z_tensor.data = z_tensor_opt
            u_tensor.data = u_tensor_opt
            z_tensor.grad.zero_()
            u_tensor.grad.zero_()

        succ_examples = []
        succ_pred_scores = []
        modif_rates = []
        adv_sentence_list = []
        if self.use_cache:
            for i in range(len(succ_discrete_z_list)):
                discrete_z = succ_discrete_z_list[i]
                discrete_u = succ_discrete_u_list[i]
                modification_rate = np.sum(discrete_z == 1) / attack_word_num
                if modification_rate > self.modification_rate:
                    continue
                modif_rates.append(modification_rate)
                adv_sentence = self.apply_substitution(discrete_z, discrete_u, idx_list, subword_idx_mat.detach().cpu().numpy())
                adv_sentence_list.append(adv_sentence)
        detached_z_tensor = z_tensor.detach().clone()[0]
        detached_u_tensor = u_tensor.detach().clone()[0]
        for i in range(self.final_sample_time):

            discrete_z = self.discretize_z(detached_z_tensor, site_mask = site_mask).detach().cpu().numpy()
            discrete_u = self.discretize_u(detached_u_tensor, site_mask = site_mask).detach().cpu().numpy()


            modification_rate = np.sum(discrete_z == 1) / attack_word_num
            if modification_rate > self.modification_rate:
                continue
            adv_sentence = self.apply_substitution(discrete_z, discrete_u, idx_list, subword_idx_mat.detach().cpu().numpy())
            adv_sentence_list.append(adv_sentence)
            modif_rates.append(modification_rate)

        if len(adv_sentence_list) == 0:
            return [],[],[], False
        
        if self.sentence_pair:
            orig_list = [sentence1 for _ in range(len(adv_sentence_list))]
            pred_prob = self.victim_model.predict(orig_list, adv_sentence_list)
        else:
            pred_prob = self.victim_model.predict(adv_sentence_list)
        orig_label_score = pred_prob[:, orig_label]
        
        pred_label = np.argmax(pred_prob, axis = -1)
        succ_idxs = np.where(pred_label != orig_label)[0]
        
        if len(succ_idxs) > 0:
            succ_examples = [adv_sentence_list[x] for x in succ_idxs]
            succ_pred_scores = pred_prob[succ_idxs]
            succ_modif_rates = [modif_rates[x] for x in succ_idxs]
            return succ_examples, succ_pred_scores, succ_modif_rates, True
        else:
            best_perturb = np.argmin(orig_label_score)

            return [adv_sentence_list[best_perturb]], [],[], False

    def attack(self, sentence, orig_label, restart_num = 10):
        if not self.sentence_pair:
            tokens, idx_list, attention_mask, token_type_ids, sentence1_tokens = self.tokenize_sentence(sentence)
            sentence_tr = self.tokenizer.convert_tokens_to_string(sentence1_tokens)
            word_list = sentence_tr.split()
            pos_list = ['none'] + pos_tag(word_list) + ['none']
            word_list = [self.tokenizer.cls_token] + word_list + [self.tokenizer.sep_token]
            attack_word_num = len(word_list[1:-1])
            self.eps = int(self.modification_rate * len(word_list[1:-1]))
        else:
            sentence1, sentence2 = sentence
            tokens, idx_list, attention_mask, token_type_ids, sentence1_tokens, sentence2_tokens = self.tokenize_sentence(sentence1, sentence2)
            sentence1_str = self.tokenizer.convert_tokens_to_string(sentence1_tokens)
            sentence2_str = self.tokenizer.convert_tokens_to_string(sentence2_tokens)
            sentence1_word_list = sentence1_str.split()
            sentence2_word_list = sentence2_str.split()
            pos_list = ['none'] + ['none'] * len(sentence1_word_list) + ['none'] + pos_tag(sentence2_word_list) + ['none']
            word_list = [self.tokenizer.cls_token] + sentence1_word_list + [self.tokenizer.sep_token] + sentence2_word_list + [self.tokenizer.sep_token]
            attack_word_num = len(sentence2_word_list)
            self.eps = int(self.modification_rate * len(sentence2_word_list))
        # import ipdb
        # ipdb.set_trace()

        if self.eps < 1:
            return [],[],[], False
        try:
            ## What is these lines of codes doing? Since we hope to skip those stopwords and only perturb nouns, verbs, adjectives, and adverbs,
            ## we need to first pos-tagging the input sentence. However, the pos-tagging is word-level instead of token-level. Regarding on those
            ## words that are tokenized into sub-words by the langauge model's tokenizer, we need to assign the POS of the original word to these
            ## sub-words. Therefore, we need to find the mapping between the original word and the subwords using the `match_subword_with_word` function

            ## some words cannot be tokenized and will cause errors when matching subwords with original words
            ## one example from RTE dataset is "CAMDEN, N.J. (Reuters) — Three Muslim brothers from Albania ...". 
            ## The ALBERT tokenizer will tokenize "—" into '⁇' and the following matching algorithm will fail.
            ## In that case, we will delete the Part-of-speech(POS) constraint for that example, and ignore the POS when attacking
            if self.victim == 'bert':
                match_index = match_subword_with_word(tokens, word_list)
            elif self.victim == 'roberta':
                match_index = match_subword_with_word_roberta(tokens, word_list, self.tokenizer)
            elif self.victim == 'albert':
                match_index = match_subword_with_word_albert(tokens, word_list)
        except:
            match_index = None
        if self.sentence_pair:
            subword_idx_mat, site_mask, sub_mask, subword_score_mat, origword_score_mat = \
                self.build_sentence_pair_neighbor_matrix(tokens, match_index, pos_list, sentence1_tokens, sentence2_tokens)        
        else:
            subword_idx_mat, site_mask, sub_mask, subword_score_mat, origword_score_mat = self.build_neighbor_matrix(tokens, match_index, pos_list)

        if self.use_lm:
            lm_loss = torch.log(subword_score_mat) - torch.log(origword_score_mat)  
        else:
            lm_loss = None


        for patience in range(restart_num):
            if self.no_subword and patience <= 1:
                ## Most pre-trained language models use byte-pair encoding. Since TextGrad perturb the sentence in token-level, it is possible that
                ## some subwords are perturbed. For example, "surprising" could be tokenized into "surpris" and "ing". Perturbing such subwords simutanesouly
                ## could lead to some grammatical errors. To relieve this problem, we mask subwords and do not perturb them in several attack trials 
                ## If the attack cannot succeed without keeping subwords unchange, then we ignore the subword constraints and attack all tokens 
                subtoken_mask = self.get_subtoken_mask(tokens)
                subtoken_mask = torch.tensor(subtoken_mask, device = self.device)
                new_site_mask = site_mask * subtoken_mask
            else:
                new_site_mask = site_mask.detach().clone()

            adv_exmaples, adv_pred_scores, adv_modif_rates, attack_flag \
                    = self.perturb(sentence, idx_list, orig_label, new_site_mask, sub_mask.detach().clone(), subword_idx_mat,
                    lm_loss, attention_mask, token_type_ids, attack_word_num)

            if attack_flag:
                break
        ## if attack succeeds, post process the adversarial examples
        if attack_flag:  
            transformed_advs = []
            if self.sentence_pair:
                for adv in adv_exmaples:
                    if sentence2[0].isupper() and adv[0].islower():    ## Captalize the first character if the first word in the original sentence is captalized.
                        recovered_str = adv[0].upper() + adv[1:]
                        transformed_advs.append(recovered_str)
                    else:
                        transformed_advs.append(adv)
            else:
                for adv in adv_exmaples:
                    if sentence[0].isupper() and adv[0].islower():
                        recovered_str = adv[0].upper() + adv[1:]
                        transformed_advs.append(recovered_str)
                    else:
                        transformed_advs.append(adv)
            adv_exmaples = transformed_advs


        return adv_exmaples, adv_pred_scores, adv_modif_rates, attack_flag

