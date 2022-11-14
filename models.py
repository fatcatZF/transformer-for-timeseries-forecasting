import math
from typing import Optional, Any, Union, Callable, Tuple
import copy

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))



"""Transformer Modules"""

"""Positional Encoding"""

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len , d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()*(-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0).transpose(0,1)
        self.register_buffer("pe", pe)
        
    def forward(self, x):
        """
        args: [seq_len, n_batch, d_model]
        """
        x = x+self.pe[:x.size(0), :]
        return self.dropout(x)


"""Time series encoder"""

class TimeSeriesEncoder(nn.Module):
    def __init__(self, n_in, d_model, dim_feedforward, nhead, num_enlayers, dropout, 
                max_len):
        super(TimeSeriesEncoder, self).__init__()
        self.fc_in = nn.Linear(n_in, d_model)
        self.pe = PositionalEncoding(d_model, dropout, max_len)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, 
                                                       dim_feedforward=dim_feedforward,
                                                       nhead=nhead, dropout=dropout), 
                                            num_layers=num_enlayers)
    def forward(self, x):
        """
        args: x, shape: [seq_len, n_batch, n_in]
        """
        x = self.fc_in(x)
        x = self.pe(x)
        return self.encoder(x)


"""Time series decoder"""

class TimeSeriesDecoder(nn.Module):
    def __init__(self, n_in, d_model, dim_feedforward, nhead, num_delayers, dropout, max_len):
        super(TimeSeriesDecoder, self).__init__()
        self.fc_in = nn.Linear(n_in, d_model)
        self.pe = PositionalEncoding(d_model, dropout=dropout, max_len=max_len)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=d_model, 
                                                       dim_feedforward=dim_feedforward,
                                                       nhead=nhead, dropout=dropout), 
                                             num_layers=num_delayers)
        self.fc_out = nn.Linear(d_model, n_in)
    
    def forward(self, x, memory, tgt_mask=None):
        """
        args:
          x: current time step, shape: [1,n_batch,n_in]
          memory: previous memory, shape: [seq_len, n_batch, d_model]
        """
        x = self.fc_in(x)
        x = self.pe(x)
        x = self.decoder(tgt=x, memory=memory, tgt_mask=tgt_mask)
        return self.fc_out(x)



"""ConvTransformer Modules"""

# temporal convolutional layer
class TCNLayer(nn.Module):
    """Temporal convolutional layer"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation, causal=False):
        super(TCNLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size= kernel_size
        if causal:
            self.padding = dilation*(kernel_size-1)
        else:
            self.padding = "same"
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                             padding=self.padding, dilation=dilation)
        self.causal = causal
        
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def forward(self, x):
        """
        shape of x: [total_seq, num_features, num_timesteps]
        """
        x = self.conv(x)
        if self.kernel_size==1:
            return x
        if not self.causal:
            return x
        else:
            return x[:,:,:-self.padding] 

# convolutional multihead attention
class ConvMultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0., batch_first=False, kernel_size=3, dilation=1,
                 causal=False):
        super(ConvMultiheadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        head_dim = d_model // num_heads
        self.head_dim = head_dim
        self.causal = causal
        assert self.head_dim * num_heads == self.d_model, "embed_dim must be divisible by num_heads"
        self.convs_Q = nn.ModuleList([TCNLayer(d_model, head_dim, kernel_size, 
                                               dilation, causal) for i in range(num_heads)])
        self.convs_K = nn.ModuleList([TCNLayer(d_model, head_dim, kernel_size, 
                                               dilation, causal) for i in range(num_heads)])
        self.convs_V = nn.ModuleList([TCNLayer(d_model, head_dim, 1, 1, causal) for i in range(num_heads)])
        self.fc_out = nn.Linear(d_model, d_model)
        
    def forward(self, query:Tensor, key:Tensor, value:Tensor, att_mask:Optional[Tensor] = None):
        """
        args:
            query,key,value; shape: [seq_len, n_batch, d_model] if batch_first==Fasle
                                    or [n_batch, seq_len, d_model] if batch_first==True
                                    
            return: Tensor: shape: [seq_len, n_batch, d_model] if batch_first==False
                                  or [n_batch, seq_len, d_model] if batch_first==True
        """
        
        """convert the shape to [n_batch, d_model, seq_len]"""
        if self.batch_first:
            query_ = query.permute(0,2,1)
            key_ = key.permute(0,2,1)
            value_ = value.permute(0,2,1)
        else:
            query_ = query.permute(1,2,0)
            key_ = key.permute(1,2,0)
            value_ = value.permute(1,2,0)
            
        seq_len = query_.size(2)
        if att_mask is None:
            att_mask = 0
        
        queries = [conv(query_) for conv in self.convs_Q] #shape:[n_batch,head_dim,seq_len]
        keys = [conv(key_) for conv in self.convs_K] #shape:[n_batch,head_dim,seq_len]
        values = [conv(value_) for conv in self.convs_V] #shape:[n_batch,head_dim,seq_len]
        attention = [F.softmax((torch.matmul(queries[i].permute(0,2,1),keys[i])/self.head_dim)+att_mask, 
                               dim=-1) for i in range(self.num_heads)] #shape:[n_batch, seq_len, seq_len]
        
        ave_att_weights = torch.stack(attention, dim=0)
        ave_att_weights = ave_att_weights.permute(1,0,2,3)
        #shape: [n_batch, n_heads, seq_len, seq_len]
        ave_att_weights = ave_att_weights.mean(dim=1)
        #shape: [n_batch, seq_len, seq_len]
        
        
        attention_values = [torch.matmul(attention[i],
                                         values[i].permute(0,2,1)) for i in range(self.num_heads)] #shape:[n_batch,seq_len, head_dim]
        attention_values = self.fc_out(torch.cat(attention_values, dim=-1))
        #shape: [n_batch,seq_len,d_model]
        if self.batch_first:
            return attention_values, ave_att_weights
        else:
            return attention_values.permute(1,0,2), ave_att_weights


# convolutional transformer encoder layer
class ConvTransformerEncoderLayer(nn.Module):
    r"""

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectively. Otherwise it's done after. Default: ``False`` (after).


    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None, kernel_size:int=3, dilation:int=1, causal:bool=False) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(ConvTransformerEncoderLayer, self).__init__()
        #self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
        #                                   **factory_kwargs)
        self.self_attn = ConvMultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                               kernel_size=kernel_size, dilation=dilation, causal=causal)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super(ConvTransformerEncoderLayer, self).__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu


    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        



        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask))
            x = self.norm2(x + self._ff_block(x))

        return x


    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           att_mask=attn_mask)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)



class ConvTransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers. Users can build the
    BERT(https://arxiv.org/abs/1810.04805) model with corresponding parameters.

    Args:
        encoder_layer: an instance of the ConvTransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
        enable_nested_tensor: if True, input will automatically convert to nested tensor
            (and convert back on output). This will improve the overall performance of
            TransformerEncoder when padding rate is high. Default: ``True`` (enabled).

    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None, enable_nested_tensor=True, mask_check=True):
        super(ConvTransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.enable_nested_tensor = enable_nested_tensor
        self.mask_check = mask_check

    def forward(self, src: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        
        output = src
        convert_to_nested = False
        first_layer = self.layers[0]
        why_not_sparsity_fast_path = ''
        str_first_layer = "self.layers[0]"
        
        for mod in self.layers:
            output = mod(output, src_mask=mask)

        if convert_to_nested:
            output = output.to_padded_tensor(0.)

        if self.norm is not None:
            output = self.norm(output)

        return output


class ConvTransformerDecoderLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None, kernel_size:int=3, dilation:int=1, 
                 causal_src:bool=False, causal_tgt:bool=False) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(ConvTransformerDecoderLayer, self).__init__()
        self.self_attn = ConvMultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            kernel_size=kernel_size, dilation=dilation,causal=causal_tgt)
        self.multihead_attn = ConvMultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            kernel_size=kernel_size, dilation=dilation, causal=causal_src)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(ConvTransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, 
                memory_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask))
            x = self.norm3(x + self._ff_block(x))

        return x


    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           att_mask=attn_mask)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor]) -> Tensor:
        x = self.multihead_attn(x, mem, mem,
                                att_mask=attn_mask)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class ConvTransformerDecoder(nn.Module):
    
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(ConvTransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None) -> Tensor:
        
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output



"""LSTM Modules"""



