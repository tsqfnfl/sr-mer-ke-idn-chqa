import numpy as np
import pandas as pd

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, base_path, task, split, tokenizer, add_ignore_label=False, *args, **kwargs):
        if task == 'sentence_recognition':
            self.LABEL2INDEX = {'B-BACKGROUND': 0, 'I-BACKGROUND': 1, 'B-QUESTION': 2, 'I-QUESTION': 3, 'B-IGNORE': 4, 'I-IGNORE': 5}
            self.INDEX2LABEL = {0: 'B-BACKGROUND', 1: 'I-BACKGROUND', 2: 'B-QUESTION', 3: 'I-QUESTION', 4: 'B-IGNORE', 5: 'I-IGNORE'}
            self.NUM_LABELS = 6
            self.task_id = 'sr'
        elif task == 'medical_entity_recognition':
            self.LABEL2INDEX = {'B-DISEASE': 0, 'I-DISEASE': 1, 'B-SYMPTOM': 2, 'I-SYMPTOM': 3, 'B-DRUG': 4, 'I-DRUG': 5, 'B-TREATMENT': 6, 'I-TREATMENT': 7, 'O': 8}
            self.INDEX2LABEL = {0: 'B-DISEASE', 1: 'I-DISEASE', 2: 'B-SYMPTOM', 3: 'I-SYMPTOM', 4: 'B-DRUG', 5: 'I-DRUG', 6: 'B-TREATMENT', 7: 'I-TREATMENT', 8: 'O'}
            self.NUM_LABELS = 9
            self.task_id = 'mer'
        elif task == 'keyphrase_extraction':
            self.LABEL2INDEX = {'B-KEYPHRASE': 0, 'I-KEYPHRASE': 1, 'O': 2}
            self.INDEX2LABEL = {0: 'B-KEYPHRASE', 1: 'I-KEYPHRASE', 2: 'O'}
            self.NUM_LABELS = 3
            self.task_id = 'ke'

        path = '{}/dataset/'.format(base_path)
        self.data = self.load_dataset(path, split)
        self.tokenizer = tokenizer
        self.add_ignore_label = add_ignore_label

    def load_dataset(self, path, split):
        data_files = {'train': 'train.csv', 'validation': 'validation.csv', 'test': 'test.csv'}
        dataset = load_dataset(path, data_files=data_files, split=split)
        dataset = dataset.map(lambda x: {
            'tokens': eval(x['tokens']), 
            '{}_tags'.format(self.task_id): [self.LABEL2INDEX[label] for label in eval(x['{}_tags'.format(self.task_id)])]
        })
        return dataset

    def __getitem__(self, index):
        data = self.data[index]
        sentence, seq_label = data['tokens'], data['{}_tags'.format(self.task_id)]
        
        # Add CLS token
        subwords = [self.tokenizer.cls_token_id]
        subword_to_word_indices = [-1]
        labels = [-100]
        
        # Add subwords
        for word_idx, word in enumerate(sentence):
            subword_list = self.tokenizer.encode(word, add_special_tokens=False)
            subwords += subword_list
            subword_to_word_indices += [word_idx for i in range(len(subword_list))]
            labels += [seq_label[word_idx]] + [-100] * (len(subword_list) - 1)
            
        # Add last SEP token
        subwords += [self.tokenizer.sep_token_id]
        subword_to_word_indices += [-1]
        labels += [-100]

        labels = labels if self.add_ignore_label else seq_label
        return np.array(subwords), np.array(subword_to_word_indices), np.array(labels), data['tokens']
    
    def __len__(self):
        return len(self.data)


class CustomDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(CustomDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn
        
    def _collate_fn(self, batch):
        batch_size = len(batch)
        max_seq_len = max(map(lambda x: len(x[0]), batch))
        max_seq_len = min(max_seq_len, 512)
        max_tag_len = max(map(lambda x: len(x[2]), batch))
        
        subword_batch = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        mask_batch = np.zeros((batch_size, max_seq_len), dtype=np.float32)
        subword_to_word_indices_batch = np.full((batch_size, max_seq_len), -1, dtype=np.int64)
        label_batch = np.full((batch_size, max_tag_len), -100, dtype=np.int64)

        tokens_list = []
        for i, (subwords, subword_to_word_indices, labels, tokens) in enumerate(batch):
            subwords = subwords[:max_seq_len]
            subword_to_word_indices = subword_to_word_indices[:max_seq_len]

            subword_batch[i, :len(subwords)] = subwords
            mask_batch[i, :len(subwords)] = 1
            subword_to_word_indices_batch[i, :len(subwords)] = subword_to_word_indices
            label_batch[i, :len(labels)] = labels
            tokens_list.append(tokens)
            
        return subword_batch, mask_batch, subword_to_word_indices_batch, label_batch, tokens_list
