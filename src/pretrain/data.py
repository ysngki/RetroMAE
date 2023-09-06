import random
from copy import deepcopy
from dataclasses import dataclass

import torch.utils.data.dataset
from datasets import Dataset
from pretrain.utils import tensorize_batch
from transformers import DataCollatorForWholeWordMask

from typing import List
from transformers import BertTokenizer, BertTokenizerFast
import warnings


class DatasetForPretraining(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.dataset = Dataset.load_from_disk(data_dir)

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)


@dataclass
class VanillaMLMCollator(DataCollatorForWholeWordMask):
    max_seq_length: int = 512
    encoder_mlm_probability: float = 0.15
    decoder_mlm_probability: float = 0.15 # no use

    def __call__(self, examples):
        input_ids_batch = []
        attention_mask_batch = []
        encoder_mlm_mask_batch = []

        tgt_len = self.max_seq_length - self.tokenizer.num_special_tokens_to_add(False)

        for e in examples:
            e_trunc = self.tokenizer.build_inputs_with_special_tokens(e['token_ids'][:tgt_len])
            tokens = [self.tokenizer._convert_id_to_token(tid) for tid in e_trunc]

            self.mlm_probability = self.encoder_mlm_probability
            text_encoder_mlm_mask = self._whole_word_mask(tokens)

            input_ids_batch.append(torch.tensor(e_trunc))
            attention_mask_batch.append(torch.tensor([1] * len(e_trunc)))
            e_trunc[0] = -100
            e_trunc[-1] = -100

            encoder_mlm_mask_batch.append(torch.tensor(text_encoder_mlm_mask))

        input_ids_batch = tensorize_batch(input_ids_batch, self.tokenizer.pad_token_id)
        attention_mask_batch = tensorize_batch(attention_mask_batch, 0)
        encoder_mlm_mask_batch = tensorize_batch(encoder_mlm_mask_batch, 0)
        encoder_input_ids_batch, encoder_labels_batch = self.torch_mask_tokens(input_ids_batch, encoder_mlm_mask_batch)

        batch = {
            "encoder_input_ids": encoder_input_ids_batch,
            "encoder_attention_mask": attention_mask_batch,
            "encoder_labels": encoder_labels_batch,
        }

        return batch
    

@dataclass
class RetroMAECollator(DataCollatorForWholeWordMask):
    max_seq_length: int = 512
    encoder_mlm_probability: float = 0.15
    decoder_mlm_probability: float = 0.15

    def _multiple_whole_word_mask(self, input_tokens: List[str], max_predictions=512, times=1):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """
        if not isinstance(self.tokenizer, (BertTokenizer, BertTokenizerFast)):
            warnings.warn(
                "DataCollatorForWholeWordMask is only suitable for BertTokenizer-like tokenizers. "
                "Please refer to the documentation for more information."
            )

        cand_indexes = []
        for i, token in enumerate(input_tokens):
            if token == "[CLS]" or token == "[SEP]":
                continue

            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

        num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * self.mlm_probability))))
        result_list = []
        for _ in range(times):
            random.shuffle(cand_indexes)
            masked_lms = []
            covered_indexes = set()
            for index_set in cand_indexes:
                if len(masked_lms) >= num_to_predict:
                    break
                # If adding a whole-word mask would exceed the maximum number of
                # predictions, then just skip this candidate.
                if len(masked_lms) + len(index_set) > num_to_predict:
                    continue
                is_any_index_covered = False
                for index in index_set:
                    if index in covered_indexes:
                        is_any_index_covered = True
                        break
                if is_any_index_covered:
                    continue
                for index in index_set:
                    covered_indexes.add(index)
                    masked_lms.append(index)

            if len(covered_indexes) != len(masked_lms):
                raise ValueError("Length of covered_indexes is not equal to length of masked_lms.")
            mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]

            result_list.append(mask_labels)
        return result_list
    

    def __call__(self, examples):
        mask_set_num = 16

        input_ids_batch = []
        attention_mask_batch = []
        encoder_mlm_mask_batch = []
        decoder_labels_batch = []
        decoder_matrix_attention_mask_batch = []

        tgt_len = self.max_seq_length - self.tokenizer.num_special_tokens_to_add(False)

        for e in examples:
            e_trunc = self.tokenizer.build_inputs_with_special_tokens(e['token_ids'][:tgt_len])
            tokens = [self.tokenizer._convert_id_to_token(tid) for tid in e_trunc]

            self.mlm_probability = self.encoder_mlm_probability
            text_encoder_mlm_mask = self._whole_word_mask(tokens)

            self.mlm_probability = self.decoder_mlm_probability

            # create different mask
            mask_set = self._multiple_whole_word_mask(tokens, times=min(len(tokens), mask_set_num))
            mask_set = torch.tensor(mask_set)

            # sample mask for each token
            idx = torch.randint(0, min(len(tokens), mask_set_num), (len(tokens), ))
            text_matrix_attention_mask = torch.index_select(mask_set, 0, idx)

            text_matrix_attention_mask = 1 - text_matrix_attention_mask

            diagnol_tensor = torch.nn.functional.one_hot(torch.arange(text_matrix_attention_mask.shape[0]), num_classes=text_matrix_attention_mask.shape[1])
            text_matrix_attention_mask = text_matrix_attention_mask * (1 - diagnol_tensor)
            
            decoder_matrix_attention_mask_batch.append(text_matrix_attention_mask)

            input_ids_batch.append(torch.tensor(e_trunc))
            attention_mask_batch.append(torch.tensor([1] * len(e_trunc)))
            e_trunc[0] = -100
            e_trunc[-1] = -100
            decoder_labels_batch.append(torch.tensor(e_trunc))

            encoder_mlm_mask_batch.append(torch.tensor(text_encoder_mlm_mask))

        input_ids_batch = tensorize_batch(input_ids_batch, self.tokenizer.pad_token_id)
        attention_mask_batch = tensorize_batch(attention_mask_batch, 0)
        origin_input_ids_batch = input_ids_batch.clone()
        encoder_mlm_mask_batch = tensorize_batch(encoder_mlm_mask_batch, 0)
        encoder_input_ids_batch, encoder_labels_batch = self.torch_mask_tokens(input_ids_batch, encoder_mlm_mask_batch)
        decoder_labels_batch = tensorize_batch(decoder_labels_batch, -100)
        matrix_attention_mask_batch = tensorize_batch(decoder_matrix_attention_mask_batch, 0)

        batch = {
            "encoder_input_ids": encoder_input_ids_batch,
            "encoder_attention_mask": attention_mask_batch,
            "encoder_labels": encoder_labels_batch,
            "decoder_input_ids": origin_input_ids_batch,
            "decoder_attention_mask": matrix_attention_mask_batch,  # [B,L,L]
            "decoder_labels": decoder_labels_batch,
        }

        return batch


@dataclass
class DupMAECollator(DataCollatorForWholeWordMask):
    max_seq_length: int = 512
    encoder_mlm_probability: float = 0.15
    decoder_mlm_probability: float = 0.15

    def __call__(self, examples):
        input_ids_batch = []
        attention_mask_batch = []
        encoder_mlm_mask_batch = []
        decoder_labels_batch = []
        decoder_matrix_attention_mask_batch = []
        bag_word_weight = []

        tgt_len = self.max_seq_length - self.tokenizer.num_special_tokens_to_add(False)

        for e in examples:
            e_trunc = self.tokenizer.build_inputs_with_special_tokens(e['token_ids'][:tgt_len])
            tokens = [self.tokenizer._convert_id_to_token(tid) for tid in e_trunc]

            self.mlm_probability = self.encoder_mlm_probability
            text_encoder_mlm_mask = self._whole_word_mask(tokens)

            self.mlm_probability = self.decoder_mlm_probability
            mask_set = []
            for _ in range(min(len(tokens), 256)):
                mask_set.append(self._whole_word_mask(tokens))

            text_matrix_attention_mask = []
            for i in range(len(tokens)):
                idx = random.randint(0, min(len(tokens), 256) - 1)
                text_decoder_mlm_mask = deepcopy(mask_set[idx])
                text_decoder_mlm_mask[i] = 1
                text_matrix_attention_mask.append(text_decoder_mlm_mask)

            input_ids_batch.append(torch.tensor(e_trunc))
            attention_mask_batch.append(torch.tensor([1] * len(e_trunc)))
            e_trunc[0] = -100
            e_trunc[-1] = -100
            decoder_labels_batch.append(torch.tensor(e_trunc))

            encoder_mlm_mask_batch.append(torch.tensor(text_encoder_mlm_mask))
            decoder_matrix_attention_mask_batch.append(1 - torch.tensor(text_matrix_attention_mask))

            weight = torch.zeros(size=(self.tokenizer.vocab_size,))
            for t in e['token_ids'][:tgt_len]:
                weight[t] = 1 / len(e['token_ids'][:tgt_len])
            bag_word_weight.append(weight.unsqueeze(0))

        input_ids_batch = tensorize_batch(input_ids_batch, self.tokenizer.pad_token_id)
        attention_mask_batch = tensorize_batch(attention_mask_batch, 0)
        origin_input_ids_batch = input_ids_batch.clone()
        encoder_mlm_mask_batch = tensorize_batch(encoder_mlm_mask_batch, 0)
        encoder_input_ids_batch, encoder_labels_batch = self.torch_mask_tokens(input_ids_batch, encoder_mlm_mask_batch)
        decoder_labels_batch = tensorize_batch(decoder_labels_batch, -100)
        matrix_attention_mask_batch = tensorize_batch(decoder_matrix_attention_mask_batch, 0)
        bag_word_weight = torch.cat(bag_word_weight, dim=0)

        batch = {
            "encoder_input_ids": encoder_input_ids_batch,
            "encoder_attention_mask": attention_mask_batch,
            "encoder_labels": encoder_labels_batch,
            "decoder_input_ids": origin_input_ids_batch,
            "decoder_attention_mask": matrix_attention_mask_batch,  # [B,L,L]
            "decoder_labels": decoder_labels_batch,
            "bag_word_weight": bag_word_weight
        }

        return batch

