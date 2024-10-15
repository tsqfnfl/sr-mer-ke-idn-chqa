import evaluate
import numpy as np
import pandas as pd
import random
import torch

from torch import optim
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig

from args_helper import get_parser
from data_utils import CustomDataset, CustomDataLoader
from token_classification import BertForTokenClassification

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def forward_token_classification(model, batch_data, i2l, tasks=['sr', 'mer', 'ke'], device='cuda', **kwargs):
    # Unpack batch data
    subword_batch, mask_batch, subword_to_word_indices_batch, label_batch = batch_data

    # Prepare input and label
    subword_batch = torch.LongTensor(subword_batch)
    mask_batch = torch.FloatTensor(mask_batch)
    subword_to_word_indices_batch = torch.LongTensor(subword_to_word_indices_batch)
    label_batch = [torch.LongTensor(label_batch[i]) for i in range(len(label_batch))]

    if device == 'cuda':
        subword_batch = subword_batch.cuda()
        mask_batch = mask_batch.cuda()
        subword_to_word_indices_batch = subword_to_word_indices_batch.cuda()
        label_batch = [label_batch[i].cuda() for i in range(len(label_batch))]

    # Forward model
    outputs = model(
        input_ids=subword_batch,
        attention_mask=mask_batch, 
        subword_to_word_ids=subword_to_word_indices_batch,
        labels=tuple(label_batch)
    )
    loss, logits = outputs[:2]

    def get_predictions(logits_batch, label_batch, i2l):
        # Generate prediction & label list
        list_predictions, list_labels = [], []
        prediction_batch = torch.topk(logits_batch, k=1, dim=-1)[1].squeeze(dim=-1)
        for i in range(len(prediction_batch)):
            predictions = prediction_batch[i].tolist()
            labels = label_batch[i].tolist()
            filtered_predictions, filtered_labels = [], []
            for j in range(len(predictions)):
                if labels[j] == -100:
                    continue
                else:
                    filtered_predictions.append(i2l[predictions[j]])
                    filtered_labels.append(i2l[labels[j]])
            list_predictions.append(filtered_predictions)
            list_labels.append(filtered_labels)
        return list_predictions, list_labels
    
    list_predictions, list_labels = tuple(), tuple()
    for i in range(len(tasks)):
        task_list_predictions, task_list_labels = get_predictions(logits[i], label_batch[i], i2l[tasks[i]])
        list_predictions += (task_list_predictions,)
        list_labels += (task_list_labels,)

    return loss, list_predictions, list_labels

def run_experiment(args):
    tasks = args['tasks'].split(',')
    experiment_name = 'mt_{}_{}'.format(
        args['structure'],
        args['pretrained_model'].split('/')[-1]
    )
    task_details = ['{}{}'.format(x, args['{}_lw'.format(x)]) for x in tasks]
    experiment_name = '{}_{}'.format(experiment_name, '_'.join(task_details))

    tokenizer = AutoTokenizer.from_pretrained(args['pretrained_model'])
    add_ignore_label = False if args['word_representation'] == 'avg' else True

    train_dataset = CustomDataset(args['base_path'], 'train', tokenizer, tasks=tasks, add_ignore_label=add_ignore_label)
    train_loader = CustomDataLoader(train_dataset, batch_size=args['batch_size'])

    validation_dataset = CustomDataset(args['base_path'], 'validation', tokenizer, tasks=tasks, add_ignore_label=add_ignore_label)
    validation_loader = CustomDataLoader(validation_dataset, batch_size=args['batch_size'])

    test_dataset = CustomDataset(args['base_path'], 'test', tokenizer, tasks=tasks, add_ignore_label=add_ignore_label)
    test_loader = CustomDataLoader(test_dataset, batch_size=args['batch_size'])

    config = AutoConfig.from_pretrained(args['pretrained_model'])
    config.tasks = tasks
    config.structure = args['structure']
    config.sr_num_labels = train_dataset.NUM_LABELS['sr']
    config.mer_num_labels = train_dataset.NUM_LABELS['mer']
    config.ke_num_labels = train_dataset.NUM_LABELS['ke']
    config.hidden_layer_dim = args['hidden_layer_dim']
    config.me_soft_emb_size_ke = args['me_soft_emb_size_ke']
    config.me_soft_emb_size_sr = args['me_soft_emb_size_sr']
    config.ke_soft_emb_size = args['ke_soft_emb_size']
    config.word_representation = args['word_representation']

    if 'bert' in args['pretrained_model']:
        model = BertForTokenClassification.from_pretrained(args['pretrained_model'], config=config)
    model.cuda()

    seqeval = evaluate.load("seqeval")
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    best_val_score = -1

    lw = [args['{}_lw'.format(task)] for task in tasks]

    for epoch in range(args['num_epochs']):
        # Train
        model.train()
        total_train_loss = 0
        list_predictions = [[] for _ in range(len(tasks))]
        list_labels = [[] for _ in range(len(tasks))]

        train_pbar = tqdm(iter(train_loader), position=0, leave=True, total=len(train_loader))
        for i, batch_data in enumerate(train_pbar):
            optimizer.zero_grad()
            loss, batch_predictions, batch_labels = forward_token_classification(model, batch_data[:-1], i2l=train_dataset.INDEX2LABEL, tasks=tasks)
            loss = sum([lw[i] * loss[i] for i in range(len(tasks))])
            
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            for i in range(len(tasks)):
                list_predictions[i] = list_predictions[i] + batch_predictions[i]
                list_labels[i] = list_labels[i] + batch_labels[i]

            train_pbar.set_description("(Epoch {}) TRAIN LOSS: {:.4f}".format((epoch+1), total_train_loss/(i+1)))

        train_eval = [seqeval.compute(predictions=list_predictions[i], references=list_labels[i])['overall_f1'] for i in range(len(tasks))]
        train_metrics = ' - '.join(['{}: {:.4f}'.format(tasks[i].upper(), train_eval[i]) for i in range(len(tasks))])
        print("(Epoch {}) TRAIN LOSS:{:.4f} - {}".format((epoch+1), total_train_loss/len(train_pbar), train_metrics))


        # Validation
        model.eval()
        total_val_loss = 0
        tokens = []
        list_predictions = [[] for _ in range(len(tasks))]
        list_labels = [[] for _ in range(len(tasks))]

        val_pbar = tqdm(iter(validation_loader), position=0, leave=True, total=len(validation_loader))
        for i, batch_data in enumerate(val_pbar): 
            batch_tokens = batch_data[-1]
            loss, batch_predictions, batch_labels = forward_token_classification(model, batch_data[:-1], i2l=validation_dataset.INDEX2LABEL, tasks=tasks)
            loss = sum([lw[i] * loss[i] for i in range(len(tasks))])

            total_val_loss += loss.item()
            tokens += batch_tokens

            assert len(batch_predictions) == len(batch_labels)
            for i in range(len(batch_predictions)):
                list_predictions[i] = list_predictions[i] + batch_predictions[i]
                list_labels[i] = list_labels[i] + batch_labels[i]

        val_eval = [seqeval.compute(predictions=list_predictions[i], references=list_labels[i])['overall_f1'] for i in range(len(tasks))]
        val_metrics = ' - '.join(['{}: {:.4f}'.format(tasks[i].upper(), val_eval[i]) for i in range(len(tasks))])
        print("(Epoch {}) VALIDATION LOSS:{:.4f} - {}".format((epoch+1), total_val_loss/len(val_pbar), val_metrics))

        val_result_df = pd.DataFrame({'tokens': tokens})
        for i in range(len(tasks)):
            task = tasks[i]
            val_result_df['{}_labels'.format(task)] = list_labels[i]
            val_result_df['{}_predictions'.format(task)] = list_predictions[i]
        val_result_df.to_csv('{}_val_last.csv'.format(experiment_name))

        # Test
        model.eval()
        total_test_loss = 0
        tokens = []
        list_predictions = [[] for _ in range(len(tasks))]
        list_labels = [[] for _ in range(len(tasks))]

        test_pbar = tqdm(iter(test_loader), position=0, leave=True, total=len(test_loader))
        for i, batch_data in enumerate(test_pbar):
            batch_tokens = batch_data[-1]        
            loss, batch_predictions, batch_labels = forward_token_classification(model, batch_data[:-1], i2l=test_dataset.INDEX2LABEL, tasks=tasks)
            loss = sum([lw[i] * loss[i] for i in range(len(tasks))])

            total_test_loss += loss.item()
            tokens += batch_tokens

            assert len(batch_predictions) == len(batch_labels)
            for i in range(len(batch_predictions)):
                list_predictions[i] = list_predictions[i] + batch_predictions[i]
                list_labels[i] = list_labels[i] + batch_labels[i]

        test_eval = [seqeval.compute(predictions=list_predictions[i], references=list_labels[i])['overall_f1'] for i in range(len(tasks))]
        test_metrics = ' - '.join(['{}: {:.4f}'.format(tasks[i].upper(), test_eval[i]) for i in range(len(tasks))])
        print("(Epoch {}) TEST LOSS:{:.4f} - {}".format((epoch+1), total_test_loss/len(test_pbar), test_metrics))
        
        test_result_df = pd.DataFrame({'tokens': tokens})
        for i in range(len(tasks)):
            task = tasks[i]
            test_result_df['{}_labels'.format(task)] = list_labels[i]
            test_result_df['{}_predictions'.format(task)] = list_predictions[i]
        test_result_df.to_csv('{}_test_last.csv'.format(experiment_name))

        current_val_score = sum(val_eval) / len(val_eval)
        if current_val_score > best_val_score:
            best_val_score = current_val_score
            val_result_df.to_csv('{}_val_best.csv'.format(experiment_name))
            test_result_df.to_csv('{}_test_best.csv'.format(experiment_name))


if __name__ == "__main__":
    args = get_parser()
    set_seed(args['seed'])
    run_experiment(args)
