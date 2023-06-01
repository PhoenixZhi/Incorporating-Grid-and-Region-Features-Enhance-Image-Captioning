from torch.nn import functional as F
from models.transformer.utils import PositionWiseFeedForward
import torch
from torch import nn
from models.transformer.attention import MultiHeadAttention, BoxMemoryAttention, GridMemoryAttention
from ..relative_embedding import BoxRelationalEmbedding, GridRelationalEmbedding, AllRelationalEmbedding

class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(EncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None,**kwargs):
        att = self.mhatt(queries, keys, values, attention_mask, attention_weights,**kwargs)
        ff = self.pwff(att)
        return ff


class MultiLevelEncoder(nn.Module):
    def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, attention_module_kwargs=None):
        super(MultiLevelEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.reg_layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                       identity_map_reordering=identity_map_reordering,
                                                       attention_module=BoxMemoryAttention,
                                                       attention_module_kwargs=attention_module_kwargs)
                                          for _ in range(N)])
        self.reg_gri = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                       identity_map_reordering=identity_map_reordering,
                                                       attention_module=GridMemoryAttention,
                                                       attention_module_kwargs=attention_module_kwargs)
                                          for _ in range(N)])
        self.padding_idx = padding_idx



    def forward(self, images,attention_mask=None, attention_weights=None):
        # input (b_s, seq_len, d_in)
        # attention_mask = (torch.sum(input, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)

        images['reg_feat'] = images['reg_feat'] + images['reg_ab']
        images['gri_feat'] = images['gri_feat'] + images['gri_ab']
        for l in self.reg_layers:
            images['reg_feat'] = l(images['reg_feat'], images['reg_feat'], images['reg_feat'], attention_mask=images['reg_mask'],
                                attention_weights=None, box=images['box'])
        images['reg_gri']=images['reg_feat']
        for l in self.reg_gri:
            images['reg_gri']=l(images['reg_gri'],images['gri_feat'],images['gri_feat'],attention_mask=images['gri_mask'],attention_weights=None, box=images['box'])


        return images


class Encoder(MultiLevelEncoder):
    def __init__(self, N, padding_idx, **kwargs):
        super(Encoder, self).__init__(N, padding_idx, **kwargs)
        self.gri_d=1024
        self.gri_fc = nn.Linear(self.gri_d, self.d_model)
        self.dropout = nn.Dropout(p=self.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, input, attention_weights=None):

        input['gri_feat'] = F.relu(self.gri_fc(input['gri_feat']))
        input['gri_feat'] = self.dropout(input['gri_feat'])
        input['gri_feat'] = self.layer_norm( input['gri_feat'])

        return super(Encoder, self).forward(input, attention_weights=attention_weights)
