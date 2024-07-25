import logging
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaModel

logger = logging.getLogger(__name__)


class CryptoLlama(LlamaConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_size = kwargs["input_size"]
        self.output_size = kwargs["output_size"]


class CryptoLlamaModel(nn.Module):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: CryptoLlama):
        super().__init__()
        self.model = LlamaModel(config)
        self.in_proj = nn.Linear(config.input_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.output_size)

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        inputs: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # inputs: (batch_size, seq_len, input_size)
        inputs = self.in_proj(inputs)  # (batch_size, seq_len, hidden_size)
        outputs = self.model(inputs_embeds=inputs, attention_mask=attention_mask)
        last_layer_hidden_states = outputs.last_hidden_state
        prediction = self.out_proj(last_layer_hidden_states)  # (batch_size, seq_len, output_size)
        return prediction

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
