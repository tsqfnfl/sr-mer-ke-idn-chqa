import numpy as np
import pandas as pd

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, base_path, split, tokenizer, tasks=['sr', 'mer', 'ke'], add_ignore_label=False, *args, **kwargs):
        self.LABEL2INDEX = {
            'sr': {'B-BACKGROUND': 0, 'I-BACKGROUND': 1, 'B-QUESTION': 2, 'I-QUESTION': 3, 'B-IGNORE': 4, 'I-IGNORE': 5},
            'mer': {'B-DISEASE': 0, 'I-DISEASE': 1, 'B-SYMPTOM': 2, 'I-SYMPTOM': 3, 'B-DRUG': 4, 'I-DRUG': 5, 'B-TREATMENT': 6, 'I-TREATMENT': 7, 'O': 8},
            'ke': {'B-KEYPHRASE': 0, 'I-KEYPHRASE': 1, 'O': 2}
        }

        self.INDEX2LABEL = {
            'sr': {0: 'B-BACKGROUND', 1: 'I-BACKGROUND', 2: 'B-QUESTION', 3: 'I-QUESTION', 4: 'B-IGNORE', 5: 'I-IGNORE'},
            'mer': {0: 'B-DISEASE', 1: 'I-DISEASE', 2: 'B-SYMPTOM', 3: 'I-SYMPTOM', 4: 'B-DRUG', 5: 'I-DRUG', 6: 'B-TREATMENT', 7: 'I-TREATMENT', 8: 'O'},
            'ke': {0: 'B-KEYPHRASE', 1: 'I-KEYPHRASE', 2: 'O'}
        }

        self.NUM_LABELS = {'sr': 6, 'mer': 9, 'ke': 3}

        path = '{}/dataset/'.format(base_path)
        self.tasks = tasks
        self.data = self.load_dataset(path, split)
        self.tokenizer = tokenizer
        self.add_ignore_label = add_ignore_label

    def load_dataset(self, path, split):
        data_files = {'train': 'train.csv', 'validation': 'validation.csv', 'test': 'test.csv'}
        dataset = load_dataset(path, data_files=data_files, split=split)
        dataset = dataset.map(lambda x: {
            'tokens': eval(x['tokens']), 
            'sr_tags': [self.LABEL2INDEX['sr'][label] for label in eval(x['sr_tags'])],
            'mer_tags': [self.LABEL2INDEX['mer'][label] for label in eval(x['mer_tags'])],
            'ke_tags': [self.LABEL2INDEX['ke'][label] for label in eval(x['ke_tags'])]
        })
        self.tasks_col = ['{}_tags'.format(task) for task in self.tasks]
        dataset = dataset.select_columns(['tokens'] + self.tasks_col)
        return dataset

    def __getitem__(self, index):
        data = self.data[index]
        sentence = data['tokens']
        tags = tuple(data[x] for x in self.tasks_col)
        
        # Add CLS token
        subwords = [self.tokenizer.cls_token_id]
        subword_to_word_indices = [-1]
        labels = [[-100] for _ in range(len(tags))]
        
        # Add subwords
        for word_idx, word in enumerate(sentence):
            subword_list = self.tokenizer.encode(word, add_special_tokens=False)
            subwords += subword_list
            subword_to_word_indices += [word_idx for i in range(len(subword_list))]
            for i in range(len(labels)):
                labels[i] = labels[i] + [tags[i][word_idx]] + [-100] * (len(subword_list) - 1)
            
        # Add last SEP token
        subwords += [self.tokenizer.sep_token_id]
        subword_to_word_indices += [-1]
        for i in range(len(labels)):
            labels[i] = labels[i] + [-100]

        labels = labels if self.add_ignore_label else tags
        labels = tuple(np.array(x) for x in labels)
        return np.array(subwords), np.array(subword_to_word_indices), labels, data['tokens']
    
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
        max_tag_len = max(map(lambda x: len(x[2][0]), batch))
        tasks = self.dataset.tasks
        
        subword_batch = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        mask_batch = np.zeros((batch_size, max_seq_len), dtype=np.float32)
        subword_to_word_indices_batch = np.full((batch_size, max_seq_len), -1, dtype=np.int64)
        label_batch = [np.full((batch_size, max_tag_len), -100, dtype=np.int64) for _ in range(len(tasks))]

        tokens_list = []
        for i, (subwords, subword_to_word_indices, labels, tokens) in enumerate(batch):
            subwords = subwords[:max_seq_len]
            subword_to_word_indices = subword_to_word_indices[:max_seq_len]

            subword_batch[i, :len(subwords)] = subwords
            mask_batch[i, :len(subwords)] = 1
            subword_to_word_indices_batch[i, :len(subwords)] = subword_to_word_indices
            for j in range(len(label_batch)):
                label_batch[j][i, :len(labels[j])] = labels[j]
            tokens_list.append(tokens)

        return subword_batch, mask_batch, subword_to_word_indices_batch, tuple(label_batch), tokens_list
