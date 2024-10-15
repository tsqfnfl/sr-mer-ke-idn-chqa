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
from token_classification import DistilBertForTokenClassification, BertForTokenClassification, XLMForTokenClassification, XLMRobertaForTokenClassification

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def forward_token_classification(model, batch_data, i2l, device='cuda', **kwargs):
    # Unpack batch data
    subword_batch, mask_batch, subword_to_word_indices_batch, label_batch = batch_data

    # Prepare input and label
    subword_batch = torch.LongTensor(subword_batch)
    mask_batch = torch.FloatTensor(mask_batch)
    subword_to_word_indices_batch = torch.LongTensor(subword_to_word_indices_batch)
    label_batch = torch.LongTensor(label_batch)

    if device == 'cuda':
        subword_batch = subword_batch.cuda()
        mask_batch = mask_batch.cuda()
        subword_to_word_indices_batch = subword_to_word_indices_batch.cuda()
        label_batch = label_batch.cuda()

    # Forward model
    outputs = model(
        input_ids=subword_batch,
        attention_mask=mask_batch, 
        subword_to_word_ids=subword_to_word_indices_batch,
        labels=label_batch
    )
    loss, logits = outputs[:2]

    # Generate prediction & label list
    list_predictions, list_labels = [], []
    prediction_batch = torch.topk(logits, k=1, dim=-1)[1].squeeze(dim=-1)
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

    return loss, list_predictions, list_labels

def run_experiment(args):
    experiment_name = 'single_{}_{}'.format(
        args['task'],
        args['pretrained_model'].split('/')[-1]
    )

    tokenizer = AutoTokenizer.from_pretrained(args['pretrained_model'])
    add_ignore_label = False if args['word_representation'] == 'avg' else True

    train_dataset = CustomDataset(args['base_path'], args['task'], 'train', tokenizer, add_ignore_label)
    train_loader = CustomDataLoader(train_dataset, batch_size=args['batch_size'])

    validation_dataset = CustomDataset(args['base_path'], args['task'], 'validation', tokenizer, add_ignore_label)
    validation_loader = CustomDataLoader(validation_dataset, batch_size=args['batch_size'])

    test_dataset = CustomDataset(args['base_path'], args['task'], 'test', tokenizer, add_ignore_label)
    test_loader = CustomDataLoader(test_dataset, batch_size=args['batch_size'])

    config = AutoConfig.from_pretrained(args['pretrained_model'])
    config.num_labels = train_dataset.NUM_LABELS
    config.word_representation = args['word_representation']

    if 'distil' in args['pretrained_model']:
        model = DistilBertForTokenClassification.from_pretrained(args['pretrained_model'], config=config)
    elif 'xlm-roberta' in args['pretrained_model']:
        model = XLMRobertaForTokenClassification.from_pretrained(args['pretrained_model'], config=config)
    elif 'xlm-mlm' in args['pretrained_model']:
        model = XLMForTokenClassification.from_pretrained(args['pretrained_model'], config=config)
    elif 'bert' in args['pretrained_model']:
        model = BertForTokenClassification.from_pretrained(args['pretrained_model'], config=config)
    model.cuda()

    seqeval = evaluate.load("seqeval")
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    best_val_score = -1

    for epoch in range(args['num_epochs']):
        # Train
        model.train()
        total_train_loss = 0
        predictions, labels = [], []

        train_pbar = tqdm(iter(train_loader), position=0, leave=True, total=len(train_loader))
        for i, batch_data in enumerate(train_pbar):
            optimizer.zero_grad()
            loss, batch_predictions, batch_labels = forward_token_classification(model, batch_data[:-1], i2l=train_dataset.INDEX2LABEL)
            
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            predictions += batch_predictions
            labels += batch_labels

            train_pbar.set_description("(Epoch {}) TRAIN LOSS: {:.4f}".format((epoch+1), total_train_loss/(i+1)))

        train_eval = seqeval.compute(predictions=predictions, references=labels)
        train_metrics = {
            "PRECISION": train_eval["overall_precision"],
            "RECALL": train_eval["overall_recall"],
            "F1": train_eval["overall_f1"],
        }

        metrics_str = ['{}: {:.4f}'.format(key, value) for key, value in train_metrics.items()]
        print("(Epoch {}) TRAIN LOSS:{:.4f} - {}".format((epoch+1), total_train_loss/len(train_pbar), ' '.join(metrics_str)))

        # Validation
        model.eval()
        total_val_loss = 0
        predictions, labels, tokens = [], [], []

        val_pbar = tqdm(iter(validation_loader), position=0, leave=True, total=len(validation_loader))
        for i, batch_data in enumerate(val_pbar): 
            batch_tokens = batch_data[-1] 
            loss, batch_predictions, batch_labels = forward_token_classification(model, batch_data[:-1], i2l=validation_dataset.INDEX2LABEL)
            total_val_loss += loss.item()

            # Calculate evaluation metrics
            predictions += batch_predictions
            labels += batch_labels
            tokens += batch_tokens

        val_eval = seqeval.compute(predictions=predictions, references=labels)
        val_metrics = {
            "PRECISION": val_eval["overall_precision"],
            "RECALL": val_eval["overall_recall"],
            "F1": val_eval["overall_f1"],
        }

        metrics_str = ['{}: {:.4f}'.format(key, value) for key, value in val_metrics.items()]
        print("(Epoch {}) VAL LOSS:{:.4f} - {}".format((epoch+1), total_val_loss/len(val_pbar), ' '.join(metrics_str)))

        val_result_df = pd.DataFrame({
            'tokens': tokens, 
            'labels': labels,
            'predictions': predictions,
        })
        val_result_df.to_csv('{}_val_last.csv'.format(experiment_name))
        
        # Test
        model.eval()
        total_test_loss = 0
        predictions, labels, tokens = [], [], []

        test_pbar = tqdm(iter(test_loader), position=0, leave=True, total=len(test_loader))
        for i, batch_data in enumerate(test_pbar):
            batch_tokens = batch_data[-1]        
            loss, batch_predictions, batch_labels = forward_token_classification(model, batch_data[:-1], i2l=test_dataset.INDEX2LABEL)
            total_test_loss += loss.item()

            # Calculate evaluation metrics
            predictions += batch_predictions
            labels += batch_labels
            tokens += batch_tokens

        test_eval = seqeval.compute(predictions=predictions, references=labels)
        test_metrics = {
            "PRECISION": test_eval["overall_precision"],
            "RECALL": test_eval["overall_recall"],
            "F1": test_eval["overall_f1"],
        }

        metrics_str = ['{}: {:.4f}'.format(key, value) for key, value in test_metrics.items()]
        print("(Epoch {}) TEST LOSS:{:.4f} - {}".format((epoch+1), total_test_loss/len(test_pbar), ' '.join(metrics_str)))

        test_result_df = pd.DataFrame({
            'tokens': tokens, 
            'labels': labels,
            'predictions': predictions,
        })
        test_result_df.to_csv('{}_test_last.csv'.format(experiment_name))

        current_val_score = val_metrics['F1']
        if current_val_score > best_val_score:
            best_val_score = current_val_score
            val_result_df.to_csv('{}_val_best.csv'.format(experiment_name))
            test_result_df.to_csv('{}_test_best.csv'.format(experiment_name))


if __name__ == "__main__":
    args = get_parser()
    set_seed(args['seed'])
    run_experiment(args)
