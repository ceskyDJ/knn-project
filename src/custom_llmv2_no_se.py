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
    """
    Base class for outputs of token classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) :
            Classification loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    start_end_logits: torch.FloatTensor = None
    parent_rels_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None



class LayoutLMv2ForCustomClassification(LayoutLMv2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.layoutlmv2 = LayoutLMv2Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.start_end_classifier = nn.Linear(config.hidden_size, 3)
        # self.par_rel1 = nn.Conv2d(1, 2)
        # self.par_rel2 = nn.Conv2d(2, 1)
        self.par_rel1 = nn.Linear(7, 50)
        self.par_rel2 = nn.Linear(50, 100)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.layoutlmv2.embeddings.word_embeddings

    # @add_start_docstrings_to_model_forward(LAYOUTLMV2_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # @replace_return_docstrings(output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
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
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.

        Returns:

        Example:

        ```python
        >>> from transformers import LayoutLMv2Processor, LayoutLMv2ForTokenClassification, set_seed
        >>> from PIL import Image
        >>> from datasets import load_dataset

        >>> set_seed(88)

        >>> datasets = load_dataset("nielsr/funsd", split="test")
        >>> labels = datasets.features["ner_tags"].feature.names
        >>> id2label = {v: k for v, k in enumerate(labels)}

        >>> processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")
        >>> model = LayoutLMv2ForTokenClassification.from_pretrained(
        ...     "microsoft/layoutlmv2-base-uncased", num_labels=len(labels)
        ... )

        >>> data = datasets[0]
        >>> image = Image.open(data["image_path"]).convert("RGB")
        >>> words = data["words"]
        >>> boxes = data["bboxes"]  # make sure to normalize your bounding boxes
        >>> word_labels = data["ner_tags"]
        >>> encoding = processor(
        ...     image,
        ...     words,
        ...     boxes=boxes,
        ...     word_labels=word_labels,
        ...     padding="max_length",
        ...     truncation=True,
        ...     return_tensors="pt",
        ... )

        >>> outputs = model(**encoding)
        >>> logits, loss = outputs.logits, outputs.loss

        >>> predicted_token_class_ids = logits.argmax(-1)
        >>> predicted_tokens_classes = [id2label[t.item()] for t in predicted_token_class_ids[0]]
        >>> predicted_tokens_classes[:5]
        ['B-ANSWER', 'B-HEADER', 'B-HEADER', 'B-HEADER', 'B-HEADER']
        ```
        """

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

        
        start_end_logits = self.start_end_classifier(sequence_output) # TODO: try using the image outputs also?

        # print(seq_length)
        # print(logits.size())
        # print(start_end_logits.size())



        # both_logits = torch.cat((logits, start_end_logits), dim=2)
        # print(both_logits.size())

        # rel_logits1 = self.par_rel1(both_logits)
        # rel_logits = self.par_rel2(rel_logits1)

        # print(rel_logits.size())



        loss = None
        if labels is not None and start_end_labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss_start_end = loss_fct(start_end_logits.view(-1, 3), start_end_labels.view(-1))
            # print(start_end_logits.view(-1, 3).size(), start_end_labels.size())
            # print(rel_logits.view(seq_length, -1).size(), parent_rels.view(-1).size())
            # loss_rel = MSELoss()(rel_logits.view(seq_length, -1).float(), parent_rels.view(-1).float())
            # loss = (loss + loss_start_end + (loss_rel/2)) / 3
            loss = (loss + loss_start_end) / 2
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
            parent_rels_logits=None,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
