import torch
from models.transformer import Transformer, Encoder, Decoder
device=torch.device('cuda')
outputs={}
outputs['gri_feat'] = torch.randn((4,60,1024)).to(device)
outputs['gri_mask'] = torch.randn((4,1,1,60)).to(device)
outputs['gri_mask']=outputs['gri_mask'].gt(0)
outputs['reg_feat'] = torch.randn((4,5,512)).to(device)
outputs['reg_mask'] = torch.randn((4,1,1,5)).to(device)
outputs['reg_mask']=outputs['reg_mask'].gt(0)
outputs['box'] = torch.randn((4,5,4)).to(device)
captions=torch.randint(0,50,(4,20)).to(device)

encoder = Encoder(3, 0,identity_map_reordering=False, attention_module_kwargs=None)
decoder = Decoder(10199, 54, 3, 1)
model = Transformer(2, encoder, decoder).to(device)
out = model(outputs, captions)
# x,y=model.beam_search(outputs, 20, 1,5, out_size=5)
print('hello')

