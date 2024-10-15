from torch import nn, stack
from transformers import DistilBertPreTrainedModel, DistilBertModel, BertPreTrainedModel, BertModel, XLMPreTrainedModel, XLMModel, XLMRobertaPreTrainedModel, XLMRobertaModel

class DistilBertForTokenClassification(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.word_representation = config.word_representation

        self.distilbert = DistilBertModel(config)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        subword_to_word_ids=None,
        labels=None,
    ):
        outputs = self.distilbert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]

        if self.word_representation == 'avg':
            max_seq_len = subword_to_word_ids.max() + 1
            word_latents = []
            for i in range(max_seq_len):
                mask = (subword_to_word_ids == i).unsqueeze(dim=-1)
                word_latents.append((sequence_output * mask).sum(dim=1) / mask.sum())
            sequence_output = stack(word_latents, dim=1)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        output = (logits,) + outputs[1:]
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            output = ((loss,) + output)
        
        return output


class BertForTokenClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.word_representation = config.word_representation

        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

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

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        output = (logits,) + outputs[2:]
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            output = ((loss,) + output)

        return output


class XLMForTokenClassification(XLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.word_representation = config.word_representation

        self.transformer = XLMModel(config)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        subword_to_word_ids=None,
        labels=None,
    ):
        outputs = self.transformer(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]

        if self.word_representation == 'avg':
            max_seq_len = subword_to_word_ids.max() + 1
            word_latents = []
            for i in range(max_seq_len):
                mask = (subword_to_word_ids == i).unsqueeze(dim=-1)
                word_latents.append((sequence_output * mask).sum(dim=1) / mask.sum())
            sequence_output = stack(word_latents, dim=1)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        output = (logits,) + outputs[1:]
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            output = ((loss,) + output)

        return output


class XLMRobertaForTokenClassification(XLMRobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.word_representation = config.word_representation

        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        subword_to_word_ids=None,
        labels=None,
    ):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]

        if self.word_representation == 'avg':
            max_seq_len = subword_to_word_ids.max() + 1
            word_latents = []
            for i in range(max_seq_len):
                mask = (subword_to_word_ids == i).unsqueeze(dim=-1)
                word_latents.append((sequence_output * mask).sum(dim=1) / mask.sum())
            sequence_output = stack(word_latents, dim=1)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        output = (logits,) + outputs[2:]
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            output = ((loss,) + output)

        return output