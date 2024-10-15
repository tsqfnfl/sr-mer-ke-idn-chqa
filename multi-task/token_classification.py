import torch.nn.functional as F

from torch import nn, stack, rand, matmul, cat
from transformers import BertPreTrainedModel, BertModel

class BertForTokenClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.tasks = config.tasks
        self.structure = config.structure
        self.num_labels = {'sr': config.sr_num_labels, 'mer': config.mer_num_labels, 'ke': config.ke_num_labels}
        self.word_representation = config.word_representation

        self.me_soft_emb_size_ke = config.me_soft_emb_size_ke
        self.me_soft_emb_size_sr = config.me_soft_emb_size_sr
        self.ke_soft_emb_size = config.ke_soft_emb_size
        self.hidden_layer_dim = config.hidden_layer_dim

        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )

        if self.structure == 'parallel':
            if 'sr' in self.tasks:
                self.sr_ffnn = nn.Linear(config.hidden_size, self.hidden_layer_dim)
                self.sr_dropout = nn.Dropout(classifier_dropout)
                self.sr_classifier = nn.Linear(self.hidden_layer_dim, self.num_labels['sr'])
            if 'mer' in self.tasks:
                self.mer_ffnn = nn.Linear(config.hidden_size, self.hidden_layer_dim)
                self.mer_dropout = nn.Dropout(classifier_dropout)
                self.mer_classifier = nn.Linear(self.hidden_layer_dim, self.num_labels['mer'])
            if 'ke' in self.tasks:
                self.ke_ffnn = nn.Linear(config.hidden_size, self.hidden_layer_dim)
                self.ke_dropout = nn.Dropout(classifier_dropout)
                self.ke_classifier = nn.Linear(self.hidden_layer_dim, self.num_labels['ke'])

        elif self.structure == 'hierarchical':
            if 'mer' in self.tasks:
                self.mer_ffnn = nn.Linear(config.hidden_size, self.hidden_layer_dim)
                self.mer_dropout = nn.Dropout(classifier_dropout)
                self.mer_classifier = nn.Linear(self.hidden_layer_dim, self.num_labels['mer'])
            if 'ke' in self.tasks:
                ke_ffnn_input_size = config.hidden_size
                if 'mer' in self.tasks:
                    self.me_soft_emb_ke = nn.Parameter(rand(self.num_labels['mer'], self.me_soft_emb_size_ke), requires_grad=True)
                    ke_ffnn_input_size += self.me_soft_emb_size_ke
                self.ke_ffnn = nn.Linear(ke_ffnn_input_size, self.hidden_layer_dim)
                self.ke_dropout = nn.Dropout(classifier_dropout)
                self.ke_classifier = nn.Linear(self.hidden_layer_dim, self.num_labels['ke'])
            if 'sr' in self.tasks:
                sr_ffnn_input_size = config.hidden_size
                if 'mer' in self.tasks:
                    self.me_soft_emb_sr = nn.Parameter(rand(self.num_labels['mer'], self.me_soft_emb_size_sr), requires_grad=True)
                    sr_ffnn_input_size += self.me_soft_emb_size_sr
                if 'ke' in self.tasks:
                    self.ke_soft_emb = nn.Parameter(rand(self.num_labels['ke'], self.ke_soft_emb_size), requires_grad=True)
                    sr_ffnn_input_size += self.ke_soft_emb_size
                self.sr_ffnn = nn.Linear(sr_ffnn_input_size, self.hidden_layer_dim)   
                self.sr_dropout = nn.Dropout(classifier_dropout)
                self.sr_classifier = nn.Linear(self.hidden_layer_dim, self.num_labels['sr'])

        else:
            raise Exception("--structure argument must be 'parallel' or 'hierarchical'")
    
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        subword_to_word_ids=None,
        labels=None,
    ):
     
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]

        if self.word_representation == 'avg':
            max_seq_len = subword_to_word_ids.max() + 1
            word_latents = []
            for i in range(max_seq_len):
                mask = (subword_to_word_ids == i).unsqueeze(dim=-1)
                word_latents.append((sequence_output * mask).sum(dim=1) / mask.sum())
            sequence_output = stack(word_latents, dim=1)

        if self.structure == 'parallel':
            if 'mer' in self.tasks:
                mer_classifier_input = F.relu(self.mer_ffnn(sequence_output))
                mer_classifier_input = self.mer_dropout(mer_classifier_input)
                mer_logits = self.mer_classifier(mer_classifier_input)
            if 'ke' in self.tasks:
                ke_classifier_input = F.relu(self.ke_ffnn(sequence_output))
                ke_classifier_input = self.ke_dropout(ke_classifier_input)
                ke_logits = self.ke_classifier(ke_classifier_input)
            if 'sr' in self.tasks:
                sr_classifier_input = F.relu(self.sr_ffnn(sequence_output))
                sr_classifier_input = self.sr_dropout(sr_classifier_input)
                sr_logits = self.sr_classifier(sr_classifier_input)

        elif self.structure == 'hierarchical':
            if 'mer' in self.tasks:
                mer_classifier_input = F.relu(self.mer_ffnn(sequence_output))
                mer_classifier_input = self.mer_dropout(mer_classifier_input)
                mer_logits = self.mer_classifier(mer_classifier_input)
            if 'ke' in self.tasks:
                ke_ffnn_input = sequence_output
                if 'mer' in self.tasks:
                    me_soft_emb_matrix_ke = self.me_soft_emb_ke.repeat(mer_logits.size(0), 1, 1)
                    me_emb_ke = matmul(mer_logits, me_soft_emb_matrix_ke)
                    ke_ffnn_input = cat((ke_ffnn_input, me_emb_ke), dim=-1) 
                ke_classifier_input = F.relu(self.ke_ffnn(ke_ffnn_input))
                ke_classifier_input = self.ke_dropout(ke_classifier_input)
                ke_logits = self.ke_classifier(ke_classifier_input)
            if 'sr' in self.tasks:
                sr_ffnn_input = sequence_output
                if 'mer' in self.tasks:
                    me_soft_emb_matrix_sr = self.me_soft_emb_sr.repeat(mer_logits.size(0), 1, 1)
                    me_emb_sr = matmul(mer_logits, me_soft_emb_matrix_sr)
                    sr_ffnn_input = cat((sr_ffnn_input, me_emb_sr), dim=-1)
                if 'ke' in self.tasks:
                    ke_soft_emb_matrix = self.ke_soft_emb.repeat(ke_logits.size(0), 1, 1)
                    ke_emb_sr = matmul(ke_logits, ke_soft_emb_matrix)
                    sr_ffnn_input = cat((sr_ffnn_input, ke_emb_sr), dim=-1)
                sr_classifier_input = F.relu(self.sr_ffnn(sr_ffnn_input))
                sr_classifier_input = self.sr_dropout(sr_classifier_input)
                sr_logits = self.sr_classifier(sr_classifier_input)

        loss, logits = tuple(), tuple()
        if 'sr' in self.tasks:
            logits += (sr_logits,)
        if 'mer' in self.tasks:
            logits += (mer_logits,)
        if 'ke' in self.tasks:
            logits += (ke_logits,)

        output = logits + outputs[2:]
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            for i in range(len(self.tasks)):
                task_loss = loss_fct(logits[i].view(-1, self.num_labels[self.tasks[i]]), labels[i].view(-1))
                loss += (task_loss,)
            output = loss, (output)

        return output
