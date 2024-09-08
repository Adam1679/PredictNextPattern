import logging
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import T5Config, T5Model
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    BaseModelOutputWithPastAndCrossAttentions,
)
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaModel

logger = logging.getLogger(__name__)


class CryptoLlama(LlamaConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_size = kwargs["input_size"]
        self.output_size = kwargs["output_size"]
        self.categorical_features = kwargs.get("categorical_features", 0)
        self.embedding_dims = kwargs.get("embedding_dims", [])
        self.num_categories = kwargs.get("num_categories", [])  # For each categorical feature
        self.dropout = kwargs.get("dropout", 0)


class CryptoLlamaModel(nn.Module):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: CryptoLlama):
        super().__init__()
        self.model = LlamaModel(config)
        self.num_categories = config.num_categories
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_cats, dim)
                for num_cats, dim in zip(config.num_categories, config.embedding_dims)
            ]
        )
        # Adjust input size to account for embeddings
        total_embedding_dim = sum(config.embedding_dims)

        assert total_embedding_dim > 0 or config.input_size > 0
        if total_embedding_dim > 0:
            self.categorical_modeling = True
        else:
            self.categorical_modeling = False

        self.in_proj = nn.Linear(config.input_size or total_embedding_dim, config.hidden_size)

        if total_embedding_dim == 0:
            self.out_proj = nn.Linear(config.hidden_size, config.output_size)
        else:
            self.out_proj = nn.ModuleList(
                [nn.Linear(config.hidden_size, num_cats) for num_cats in config.num_categories]
            )
        self.config = config
        if config.dropout > 0:
            self.dropout = nn.Dropout(config.dropout)  # Add dropout layer
        else:
            self.dropout = None
        self._num_parameters = None

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        inputs: Optional[Union[torch.FloatTensor, torch.LongTensor]] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # inputs: (batch_size, seq_len, input_size)

        if self.categorical_modeling:
            embedded = [emb(inputs[:, :, i]) for i, emb in enumerate(self.embeddings)]
            inputs = torch.cat(embedded, dim=-1)

        inputs = self.in_proj(inputs)  # (batch_size, seq_len, hidden_size)
        if self.dropout:
            inputs = self.dropout(inputs)  # Apply dropout after input projection
        position_ids = torch.cumsum(attention_mask, dim=1)
        outputs = self.model(
            inputs_embeds=inputs, attention_mask=attention_mask, position_ids=position_ids
        )
        last_layer_hidden_states = outputs.last_hidden_state
        if not self.categorical_modeling:
            prediction = self.out_proj(
                last_layer_hidden_states
            )  # (batch_size, seq_len, output_size)
            return prediction
        else:
            predictions = [
                proj(last_layer_hidden_states) for proj in self.out_proj
            ]  # [(batch_size, seq_len, n_classes), ...]
            return predictions

    def num_parameters(self):
        if self._num_parameters is None:
            self._num_parameters = sum(p.numel() for p in self.parameters())
        return self._num_parameters

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.num_parameters()
        cfg = self.config
        L, H, Q, T = (
            cfg.num_hidden_layers,
            cfg.num_attention_heads,
            cfg.hidden_size // cfg.num_attention_heads,
            cfg.max_position_embeddings,
        )
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised_h100 = 989e12  # H100 GPU bfloat16 peak flops is 989 TFLOPS
        mfu = flops_achieved / flops_promised_h100
        return mfu


class CryptoT5Config(T5Config):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_size = kwargs["input_size"]
        self.output_size = kwargs["output_size"]
        self.categorical_features = kwargs.get("categorical_features", 0)
        self.embedding_dims = kwargs.get("embedding_dims", [])
        self.num_categories = kwargs.get("num_categories", [])  # For each categorical feature


class CryptoT5Model(nn.Module):
    """
    T5-based model for cryptocurrency prediction.

    Args:
        config: CryptoT5Config
    """

    def __init__(self, config: CryptoT5Config):
        super().__init__()
        self.model = T5Model(config)
        self.num_categories = config.num_categories
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_cats, dim)
                for num_cats, dim in zip(config.num_categories, config.embedding_dims)
            ]
        )
        # Adjust input size to account for embeddings
        total_embedding_dim = sum(config.embedding_dims)

        assert total_embedding_dim > 0 or config.input_size > 0
        if total_embedding_dim > 0:
            self.categorical_modeling = True
        else:
            self.categorical_modeling = False

        self.in_proj = nn.Linear(config.input_size or total_embedding_dim, config.d_model)

        if total_embedding_dim == 0:
            self.out_proj = nn.Linear(config.d_model, config.output_size)
        else:
            self.out_proj = nn.ModuleList(
                [nn.Linear(config.d_model, num_cats) for num_cats in config.num_categories]
            )
        self.config = config
        self._num_parameters = None

    def forward(
        self,
        attention_mask: torch.Tensor,
        decoder_inputs: torch.LongTensor,
        inputs: torch.LongTensor,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        # inputs: (batch_size, seq_len, input_size)

        if self.categorical_modeling:
            embedded = [emb(inputs[:, :, i]) for i, emb in enumerate(self.embeddings)]
            inputs = torch.cat(embedded, dim=-1)

            decoder_embedded = [
                emb(decoder_inputs[:, :, i]) for i, emb in enumerate(self.embeddings)
            ]
            decoder_inputs = torch.cat(decoder_embedded, dim=-1)
        inputs = self.in_proj(inputs)  # (batch_size, seq_len, d_model)
        decoder_inputs = self.in_proj(decoder_inputs)  # (batch_size, seq_len, d_model)
        # Use inputs as encoder_hidden_states
        outputs = self.model(
            inputs_embeds=inputs,
            decoder_inputs_embeds=decoder_inputs,
            attention_mask=attention_mask,
        )
        decoder_last_hidden_state = outputs[0]
        if not self.categorical_modeling:
            prediction = self.out_proj(
                decoder_last_hidden_state
            )  # (batch_size, seq_len, output_size)
            return prediction
        else:
            predictions = [
                proj(decoder_last_hidden_state) for proj in self.out_proj
            ]  # [(batch_size, seq_len, n_classes), ...]
            predictions = [torch.clamp(pred, -10.0, 10.0) for pred in predictions]
            return predictions

    def num_parameters(self):
        if self._num_parameters is None:
            self._num_parameters = sum(p.numel() for p in self.parameters())
        return self._num_parameters

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.num_parameters()
        cfg = self.config
        L, H, Q, T = (
            cfg.num_layers,
            cfg.num_heads,
            cfg.d_kv,
            cfg.max_position_embeddings,
        )
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised_h100 = 989e12  # H100 GPU bfloat16 peak flops is 989 TFLOPS
        mfu = flops_achieved / flops_promised_h100
        return mfu
