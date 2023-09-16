import logging

import torch
from pretrain.arguments import ModelArguments
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertForMaskedLM, AutoModelForMaskedLM, BertPreTrainedModel, BertModel
from transformers.modeling_outputs import MaskedLMOutput
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, add_code_sample_docstrings
from transformers.activations import ACT2FN
from typing import List, Optional, Tuple, Union, Mapping, Any

logger = logging.getLogger(__name__)


class BERTForPretraining(nn.Module):
    def __init__(
            self,
            bert,
            model_args: ModelArguments,
    ):
        super(BERTForPretraining, self).__init__()
        self.lm = bert

        self.model_args = model_args

    def forward(self,
                encoder_input_ids, encoder_attention_mask, encoder_labels):
        # return (torch.sum(self.lm.bert.embeddings.position_ids[:, :decoder_input_ids.size(1)]), )
        lm_out: MaskedLMOutput = self.lm(
            encoder_input_ids, encoder_attention_mask,
            labels=encoder_labels,
            output_hidden_states=True,
            return_dict=True
        )

        return (lm_out.loss,)

    def save_pretrained(self, output_dir: str):
        self.lm.save_pretrained(output_dir)
    
    def load_state_dict(self, state_dict: Mapping[str, Any],
                        strict: bool = True):
        r"""Copies parameters and buffers from :attr:`state_dict` into
        this module and its descendants. If :attr:`strict` is ``True``, then
        the keys of :attr:`state_dict` must exactly match the keys returned
        by this module's :meth:`~torch.nn.Module.state_dict` function.

        Args:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            strict (bool, optional): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~torch.nn.Module.state_dict` function. Default: ``True``

        Returns:
            ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
                * **missing_keys** is a list of str containing the missing keys
                * **unexpected_keys** is a list of str containing the unexpected keys

        Note:
            If a parameter or buffer is registered as ``None`` and its corresponding key
            exists in :attr:`state_dict`, :meth:`load_state_dict` will raise a
            ``RuntimeError``.
        """
        return self.lm.load_state_dict(state_dict, strict)

    @classmethod
    def from_pretrained(
            cls, model_args: ModelArguments,
            *args, **kwargs
    ):
        hf_model = AutoModelForMaskedLM.from_pretrained(*args, **kwargs)
        model = cls(hf_model, model_args)
        return model
    

class YYHBERTForPretraining(nn.Module):
    def __init__(
            self,
            bert,
            model_args: ModelArguments,
    ):
        super(YYHBERTForPretraining, self).__init__()
        self.lm = bert

        self.model_args = model_args

    def forward(self,
                encoder_input_ids, encoder_attention_mask, encoder_labels):
        # return (torch.sum(self.lm.bert.embeddings.position_ids[:, :decoder_input_ids.size(1)]), )
        lm_out: MaskedLMOutput = self.lm(
            encoder_input_ids, encoder_attention_mask,
            labels=encoder_labels,
            output_hidden_states=True,
            return_dict=True
        )

        return (lm_out.loss,)

    def save_pretrained(self, output_dir: str):
        self.lm.save_pretrained(output_dir, variant='lm')

    @classmethod
    def from_pretrained(
            cls, model_args: ModelArguments,
            *args, **kwargs
    ):
        hf_model = YYHBertForMaskedLM.from_pretrained(*args, **kwargs)
        model = cls(hf_model, model_args)
        return model
    
    # def load_state_dict(self, state_dict: Mapping[str, Any],
    #                     strict: bool = True):
    #     r"""Copies parameters and buffers from :attr:`state_dict` into
    #     this module and its descendants. If :attr:`strict` is ``True``, then
    #     the keys of :attr:`state_dict` must exactly match the keys returned
    #     by this module's :meth:`~torch.nn.Module.state_dict` function.

    #     Args:
    #         state_dict (dict): a dict containing parameters and
    #             persistent buffers.
    #         strict (bool, optional): whether to strictly enforce that the keys
    #             in :attr:`state_dict` match the keys returned by this module's
    #             :meth:`~torch.nn.Module.state_dict` function. Default: ``True``

    #     Returns:
    #         ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
    #             * **missing_keys** is a list of str containing the missing keys
    #             * **unexpected_keys** is a list of str containing the unexpected keys

    #     Note:
    #         If a parameter or buffer is registered as ``None`` and its corresponding key
    #         exists in :attr:`state_dict`, :meth:`load_state_dict` will raise a
    #         ``RuntimeError``.
    #     """
    #     return self.lm.load_state_dict(state_dict, strict)


######################################################################################
######################################################################################
##### following is bert copied from transformers
_CHECKPOINT_FOR_DOC = "bert-base-uncased"
_CONFIG_FOR_DOC = "BertConfig"

BERT_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`BertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""
BERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states
    

# class BertLMPredictionHead(nn.Module):
#     def __init__(self, config, emb_num):
#         super().__init__()
#         self.emb_num = emb_num
#         self.vocab_size = config.vocab_size
#         self.hidden_size = config.hidden_size

#         self.transform = BertPredictionHeadTransform(config)

#         self.projector = nn.Linear(config.hidden_size, config.hidden_size  * emb_num)

#         # The output weights are the same as the input embeddings, but there is
#         # an output-only bias for each token.
#         self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

#         self.bias = nn.Parameter(torch.zeros(config.vocab_size))

#         # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
#         self.decoder.bias = self.bias

#     def forward(self, hidden_states):
#         hidden_states = self.projector(hidden_states)
#         target_shape = list(hidden_states.shape)[:-1]
#         target_shape.extend([self.emb_num, self.hidden_size])
#         hidden_states = hidden_states.reshape(target_shape)

#         hidden_states = self.transform(hidden_states)
#         hidden_states = self.decoder(hidden_states)

#         hidden_states = torch.max(hidden_states, dim=-2)[0].contiguous()

#         return hidden_states



class BertLMPredictionHead(nn.Module):
    def __init__(self, config, emb_num, code_num):
        super().__init__()
        self.emb_num = emb_num
        self.code_num = code_num
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        self.transform = BertPredictionHeadTransform(config)

        self.code_book = nn.Linear(config.hidden_size, self.code_num, bias=False)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.voc_mapping = nn.Linear(self.code_num, config.vocab_size * self.emb_num, bias=False)

        self.voc_bias = nn.Parameter(torch.zeros(config.vocab_size * self.emb_num))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.voc_mapping.bias = self.voc_bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.code_book(hidden_states)
        hidden_states = self.voc_mapping(hidden_states)

        target_shape = list(hidden_states.shape)[:-1]
        target_shape.extend([self.vocab_size, self.emb_num])
        hidden_states = hidden_states.reshape(target_shape)

        hidden_states = torch.max(hidden_states, dim=-1)[0].contiguous()

        return hidden_states
    

class BertOnlyMLMHead(nn.Module):
    def __init__(self, config, emb_num, code_num):
        super().__init__()
        self.predictions = BertLMPredictionHead(config, emb_num=emb_num, code_num=code_num)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores
    

@add_start_docstrings("""Bert Model with a `language modeling` head on top.""", BERT_START_DOCSTRING)
class YYHBertForMaskedLM(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"predictions.decoder.bias", r"cls.predictions.decoder.weight", r"cls.predictions.bias"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"cls.predictions.code_book.weight", r"predictions.voc_mapping.bias", r"cls.predictions.voc_mapping.weight", r"cls.predictions.voc_bias"]

    def __init__(self, config, emb_num=1, code_num=1):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config, emb_num=emb_num, code_num=code_num)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return None

    def set_output_embeddings(self, new_embeddings):
        raise Exception("yyh here!!!")
        self.cls.predictions.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="'paris'",
        expected_loss=0.88,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        if self.config.pad_token_id is None:
            raise ValueError("The PAD token should be defined for generation")

        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}