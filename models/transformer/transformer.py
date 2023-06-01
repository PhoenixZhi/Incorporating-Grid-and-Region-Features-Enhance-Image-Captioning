import torch
from torch import nn
import copy
from models.containers import ModuleList
from ..captioning_model import CaptioningModel
from .. import position_encoding as pe

class Transformer(CaptioningModel):
    def __init__(self, bos_idx, encoder, decoder):
        super(Transformer, self).__init__()
        self.bos_idx = bos_idx
        self.encoder = encoder
        self.decoder = decoder
        self.register_state('reg_feat', None)
        self.register_state('reg_mask', None)
        self.register_state('reg_gri', None)
        self.box_embedding = nn.Linear(4, 512)
        self.grid_embedding = pe.PositionEmbeddingSine(256, normalize=True)
        self.init_weights()

    @property
    def d_model(self):
        return self.decoder.d_model

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def get_pos_embedding(self, boxes, split=False, gri_mask=None):
        bs = boxes.shape[0]
        region_embed = self.box_embedding(boxes)
        grid_embed = self.grid_embedding(mask=gri_mask)
        # if not self.args.box_embed:
        #     # print('reach here')
        #     region_embed = torch.zeros_like(region_embed)
        # if not self.args.grid_embed:
        #     # print('reach here')
        #     grid_embed = torch.zeros_like(grid_embed)
        if not split:
            pos = torch.cat([region_embed, grid_embed], dim=1)
            return pos
        else:
            return region_embed, grid_embed

    def forward(self, images, seq, *args):
        images['reg_ab'], images['gri_ab'] = self.get_pos_embedding(images['box'], split=True, gri_mask=images['gri_mask'].view(-1, 6, 10))
        visual_input = self.encoder(images)
        dec_output = self.decoder(seq, visual_input)
        return dec_output

    def init_state(self, b_s, device):
        return [torch.zeros((b_s, 0), dtype=torch.long, device=device),
                None, None]

    def step(self, t, prev_output, images, seq, mode='teacher_forcing', **kwargs):
        it = None
        if mode == 'teacher_forcing':
            raise NotImplementedError
        elif mode == 'feedback':
            if t == 0:
                images['reg_ab'], images['gri_ab'] = self.get_pos_embedding(images['box'], split=True,
                                                                            gri_mask=images['gri_mask'].view(-1, 6, 10))
                vis_output = self.encoder(images)
                self.reg_feat, self.reg_mask, self.reg_gri= vis_output['reg_feat'],vis_output['reg_mask'],vis_output['reg_gri']
                if isinstance(images, torch.Tensor):
                    it = images.data.new_full((images.shape[0], 1), self.bos_idx)
                else:
                    it = images['reg_feat'].data.new_full((images['reg_feat'].shape[0], 1), self.bos_idx)
            else:
                vis_output={}
                it = prev_output
                vis_output['reg_feat'],vis_output['reg_mask'],vis_output['reg_gri']=self.reg_feat,self.reg_mask,self.reg_gri

        return self.decoder(it, vis_output)


class TransformerEnsemble(CaptioningModel):
    def __init__(self, model: Transformer, weight_files):
        super(TransformerEnsemble, self).__init__()
        self.n = len(weight_files)
        self.models = ModuleList([copy.deepcopy(model) for _ in range(self.n)])
        for i in range(self.n):
            state_dict_i = torch.load(weight_files[i])['state_dict']
            self.models[i].load_state_dict(state_dict_i)

    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        out_ensemble = []
        for i in range(self.n):
            out_i = self.models[i].step(t, prev_output, visual, seq, mode, **kwargs)
            out_ensemble.append(out_i.unsqueeze(0))

        return torch.mean(torch.cat(out_ensemble, 0), dim=0)
