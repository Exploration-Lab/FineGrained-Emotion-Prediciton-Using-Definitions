import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel

tasks = [28,2]
class BertForMultiLabelClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 28)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)
        self.loss_fct1 = nn.CrossEntropyLoss()
        self.loss_fct0 = nn.BCEWithLogitsLoss()

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            task_id=None
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        pooled_output = outputs[1]
        if task_id==0:
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)

            outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

            if labels is not None:
                loss = self.loss_fct0(logits, labels)
                outputs = (loss,) + outputs
        else:
            seq_relationship_scores = self.seq_relationship(pooled_output)
            outputs = (seq_relationship_scores,) + outputs[2:]  # add hidden states and attention if they are here
            if labels is not None:
                next_sentence_loss = self.loss_fct1(seq_relationship_scores.view(-1,2), labels.view(-1))
                #loss = self.loss_fct(logits, labels)
                outputs = (next_sentence_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
