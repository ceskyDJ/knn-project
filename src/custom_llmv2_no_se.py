from transformers.utils import ModelOutput

from dataclasses import dataclass
from transformers import LayoutLMv2Processor, LayoutLMv2ImageProcessor
from transformers import LayoutLMv2PreTrainedModel, LayoutLMv2Model
from transformers.modeling_outputs import TokenClassifierOutput
from typing import Optional, Tuple, Union
import torch

from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss



@dataclass
class TokenClassifier2Output(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    start_end_logits: torch.FloatTensor = None
    blob_logits: torch.FloatTensor = None
    parent_rels_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None



class LayoutLMv2ForCustomClassification(LayoutLMv2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.layoutlmv2 = LayoutLMv2Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.classifier = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier2 = nn.Linear(config.hidden_size, config.num_labels)

        # self.start_end_classifier = nn.Linear(config.hidden_size, 3)

        self.start_end_classifier = nn.Linear(config.hidden_size, config.hidden_size)
        self.start_end_classifier2 = nn.Linear(config.hidden_size, 3)
        # self.par_rel1 = nn.Conv2d(1, 2)
        # self.par_rel2 = nn.Conv2d(2, 1)
        # self.par_rel1 = nn.Linear(7, 50)
        # self.par_rel2 = nn.Linear(50, 100)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.layoutlmv2.embeddings.word_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        bbox: Optional[torch.LongTensor] = None,
        image: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        start_end_labels: Optional[torch.LongTensor] = None, # NOTE: Added
        parent_rels: Optional[torch.LongTensor] = None, # NOTE: Added
        blob_labels: Optional[torch.LongTensor] = None, # NOTE: Added
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.layoutlmv2(
            input_ids=input_ids,
            bbox=bbox,
            image=image,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        # only take the text part of the output representations
        sequence_output = outputs[0][:, :seq_length]
        sequence_output = self.dropout(sequence_output)

        # print(sequence_output.size())

        logits = self.classifier(sequence_output)
        logits = self.classifier2(logits)

        
        blob_logits = None
        start_end_logits = None

        # # Use for SE configuration
        # start_end_logits = self.start_end_classifier(sequence_output)
        # start_end_logits = self.start_end_classifier2(start_end_logits)

        # Use for BLOB configuration
        blob_logits = self.start_end_classifier(sequence_output)
        blob_logits = self.start_end_classifier2(blob_logits)



        loss = None
        if labels is not None and start_end_labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss_start_end = loss_fct(start_end_logits.view(-1, 3), start_end_labels.view(-1))
            loss = (loss + loss_start_end) / 2

        elif labels is not None and blob_labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss_blob = loss_fct(blob_logits.view(-1, 3), blob_labels.view(-1))
            loss = (0.3 * loss + 0.7 * loss_blob)

        elif labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifier2Output(
            loss=loss,
            logits=logits,
            start_end_logits=start_end_logits,
            blob_logits=blob_logits,
            parent_rels_logits=None,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
