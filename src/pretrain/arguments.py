from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataTrainingArguments:
    data_dir: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrain data"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated. Default to the max input length of the model."
        },
    )
    encoder_mlm_probability: float = field(default=0.3)
    decoder_mlm_probability: float = field(default=0.5)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    pretrain_method: Optional[str] = field(
        default=None, metadata={"help": "Pretrain method: retromae or dupmae"}
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
                    "Don't set if you want to train a model from scratch."
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )

    bow_loss_weight: Optional[float] = field(
        default=0.1, metadata={"help": "The weight of bag_of_word_loss"}
    )

    emb_num: Optional[int] = field(
        default=1, metadata={"help": "how many embeddings per token of decoder"}
    )

    code_num: Optional[int] = field(
        default=1, metadata={"help": "when factorize the embedding, we have two matrix as (vocab_size, code_num) and (code num, hidden states)"}
    )

    label_smoothing: Optional[float] = field(
        default=0.0, metadata={"help": "Use for cross entropy"}
    )
