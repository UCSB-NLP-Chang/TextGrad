import numpy as np
class LogModule():
    def __init__(self):
        self.log_data = []

    def record(self, result, sample_index, orig_sample, adv_sample, orig_label, adv_label, invoke_time = 0):
        sample = {
            'result': result,
            'sample_index':sample_index,
            'orig_sample':orig_sample,
            'adv_sample':adv_sample,
            'orig_label':orig_label,
            'adv_label':adv_label,
            'invoke_time': invoke_time,
        }
        self.log_data.append(sample)
    