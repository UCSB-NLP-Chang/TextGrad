# TextGrad
This is the official implementation of the paper *TextGrad: Advancing Robustness Evaluation in NLP by Gradient-Driven Optimization*.

### Requirements
The dependency packages can be found in `requirements.txt` file. One can use `pip install -r requirements.txt` to configure the environment. We use python 3.7 to run the experiments.

### (Optional) Fine-tuning Language Models for Downstream Tasks
For standard training, one may use the following scripts to fine-tune a pre-trained transformer encoder model:
```sh
Model=bert    ## bert/roberta/albert
Dataset=sst   ## sst/mnli/qnli/rte/agnews
CUDA_VISIBLE_DEVICES=0 python finetuning.py --dataset $Dataset --model $Model --do_train --do_eval --learning_rate 2e-5  --evaluation_strategy epoch --num_train_epochs 3 --save_strategy epoch --save_total_limit 2  --logging_strategy epoch --output_dir ./checkpoints/finetune/$Dataset-$Model/ --load_best_model_at_end --metric_for_best_model eval_acc --disable_tqdm 0 --prediction_loss_only 0  --per_device_train_batch_size 16 --report_to wandb
```
For robust models, we use the official implementations from [here](https://github.com/RockyLzy/TextDefender).

### Attacking a Fine-tuned Model
Use `run_attack.py ` to attack a fine-tuned model. One may either fine-tune a model for attacking or use the online (from Huggingface) fine-tuned models. For example, to attack the BERT classifier on SST dataset using the checkpoint released by [TextAttack](https://github.com/QData/TextAttack), we can use the following scripts:
```{sh}
python run_attack.py --dataset sst --victim bert --model_name_or_path textattack/bert-base-uncased-SST-2 --rand --use_lm --ste --norm --iter_time 20 --cw --multi_sample --patience 10 --size 100  --modif 0.25
```

Explanations of the parameters:

`dataset`: the dataset for robustness evaluation (sst/mnli/qnli/rte/agnews)
`victim`: indicate the type of transformer-based language models you use (bert/roberta/albert)
`model_name_or_path`: the model name (online Huggingface models) or the path to the fine-tuned models in your local directory.
`rand`: random initialize the variable $z$ and variable $u$ before optimization.
`use_lm`: use masked language model loss for regularization
`ste`: use the straight-through estimator for gradient calculation
`norm`: normalize the gradients before the update of $z$ and $u$
`iter_time`: number of PGD iterations 
`cw`: use C\&W attack loss instead of CE loss
`patience`: number of allowed attack restart times. 
`size`: number of examples to attack
`modif`: attack budget for word modification.

Similarly, to attack sentence pair classification tasks (*e.g.*, MNLI dataset), one can use the following scripts:

```sh
python run_attack.py --dataset mnli --victim bert --model_name_or_path textattack/bert-base-uncased-MNLI --rand --use_lm --ste --norm --iter_time 20 --cw --multi_sample --patience 10 --size 100 --modif 0.25
```

By default, the attack log file will be saved to `attack_log/textgrad_{dataset}_{victim}.pkl`. 



After the attack finished, one can evaluate and export the adversarial examples using the following scripts:

```sh
python evaluate.py --victim bert --dataset sst --log_path /YourPath/to/log.pkl
```

The command parameter `log_path` should be the path to the corresponding  attack log file



### Robust Training
To run adversarial training (AT) using TextGrad, one can use the following scripts.
```sh
python robust_training --dataset sst --victim bert --rand --use_lm --ste --norm --iter_time 5 --cw --patience 1 --multi_sample --modif 0.25
```
By default, the adversarial training follows the [PGD-AT](https://arxiv.org/abs/1706.06083) method. To use TRADES for adversarial training, one can add `--trades` in the above scripts.

### Acknowledgment
Some of our experiments are based on [TextAttack](https://github.com/QData/TextAttack) toolkits and the repository of [TextDefender](https://github.com/RockyLzy/TextDefender):
```tex
@inproceedings{morris2020textattack,
  title={TextAttack: A Framework for Adversarial Attacks, Data Augmentation, and Adversarial Training in NLP},
  author={Morris, John and Lifland, Eli and Yoo, Jin Yong and Grigsby, Jake and Jin, Di and Qi, Yanjun},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations},
  year={2020}
}

@inproceedings{li2021searching,
  title={Searching for an Effective Defender: Benchmarking Defense against Adversarial Word Substitution},
  author={Li, Zongyi and Xu, Jianhan and Zeng, Jiehang and Li, Linyang and Zheng, Xiaoqing and Zhang, Qi and Chang, Kai-Wei and Hsieh, Cho-Jui},
  booktitle={Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  year={2021}
}
```

### Citation
```tex
@article{hou2022textgrad,
  title={TextGrad: Advancing Robustness Evaluation in NLP by Gradient-Driven Optimization},
  author={Hou, Bairu and Jia, Jinghan and Zhang, Yihua and Zhang, Guanhua and Zhang, Yang and Liu, Sijia and Chang, Shiyu},
  journal={arXiv preprint arXiv:2212.09254},
  year={2022}
}
```

