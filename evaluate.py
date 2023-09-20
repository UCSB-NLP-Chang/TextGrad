import os
from src.eval.eval_util import GPT2_processor
from src.data_util.dataloader import get_task_type
import pickle
import numpy as np
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--victim', type = str, default = 'bert')
parser.add_argument('--log_path', type = str, default = 'attack_log/textgrad_sst.pkl')
parser.add_argument('--dataset', type = str, default = 'sst')

args = parser.parse_args()

def reduplicate_list(item_list):
    filtered_list = []
    for item in item_list:
        if item not in filtered_list:
            filtered_list.append(item)
    return filtered_list

lm_model = GPT2_processor(cache_dir = './model_cache/gpt2_model/gpt2_xl/')

with open(args.log_path, 'rb') as f:
    logger = pickle.load(f)
print("evaluating: ", args.log_path)

sentence_pair = get_task_type(args.dataset)

log_data = logger.log_data

adv_sample_list = [x['adv_sample'] for x in log_data]
orig_sample_list = [x['orig_sample'] for x in log_data]
orig_labels = [x['orig_label'] for x in log_data]
result_list = [x['result'] for x in log_data]

adv_multiple_sentences = adv_sample_list[:]
orig_sentences = orig_sample_list

ppl_by_sample = []
adv_by_sample = []
orig_by_sample = []
skip_ppl = []
success = 0
fail = 0
for i in tqdm(range(len(adv_multiple_sentences))):
    curr_adv_list = adv_multiple_sentences[i]
    if sentence_pair:
        curr_orig = orig_sentences[i][1]    
    else:
        curr_orig = orig_sentences[i]
    curr_result = result_list[i]
    if not curr_result:
        fail += 1
        adv_by_sample.append("*fail*")
        continue
    unique_adv_list = reduplicate_list(curr_adv_list)
    filtered_advs = []
    adv_ppls = []
    orig_by_sample.append(orig_sentences[i])
    for adv_x in unique_adv_list:
        ppl = lm_model.predict_ppl(adv_x)
        adv_ppls.append(ppl)
        filtered_advs.append(adv_x)

    min_ppl_idx = np.argmin(adv_ppls)
    min_ppl_sample = filtered_advs[min_ppl_idx]
    min_ppl = adv_ppls[min_ppl_idx]
    adv_by_sample.append(min_ppl_sample)
    ppl_by_sample.append(min_ppl)
    success += 1



print("success: ", success)
print("fail: ",fail)
print("attack success rate: ", success / (success + fail))
print("avg ppl: ", np.mean(ppl_by_sample))

os.makedirs("adv_examples", exist_ok=True)

save_name = f"adv_examples/attack_{args.victim}_{args.dataset}.txt"
assert len(adv_by_sample) == len(orig_sentences)
with open(save_name, 'w', encoding = 'utf-8') as f:
    for i in range(len(adv_by_sample)):
        orig = orig_sentences[i]
        adv = adv_by_sample[i]
        label = orig_labels[i]
        if not sentence_pair:
            f.write("original: " + orig + '\n')
            f.write("attacked: " + adv + '\n')
        else:
            f.write("original premise   : " + orig[0] + '\n')
            f.write("original hypothesis: " + orig[1] + '\n')
            f.write("attacked hypothesis: " + adv + '\n')

        f.write("orig label: " + str(label) + '\n')
        f.write('\n')
