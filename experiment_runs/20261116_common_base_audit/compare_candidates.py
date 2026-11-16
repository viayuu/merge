"""Compare checkpoint deltas and the CUDA-device-dependent initialization coordinates."""
import json
from pathlib import Path
import sys

import torch

RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
sys.path.insert(0,str(ROOT/'URS'))
from model.Model import Model

params=dict(embedding_dim=128,encoder_layer_num=12,ff_hidden_dim=512,
            logit_clipping=50,demand_max1=True,eval_type='greedy')
model=Model(**params)
names=[k for k,_ in model.named_parameters()]
a=torch.load(RUN/'init_4090_seed3407.pt',map_location='cpu',weights_only=True)['model_state_dict']
b=torch.load(RUN/'init_3090_seed3407.pt',map_location='cpu',weights_only=True)['model_state_dict']
keys=list(a)
changed={k:a[k]!=b[k] for k in keys if not torch.equal(a[k],b[k])}
veca=torch.cat([a[k].flatten() for k in keys])
vecb=torch.cat([b[k].flatten() for k in keys])
subseta=torch.cat([a[k][mask] for k,mask in changed.items()])
subsetb=torch.cat([b[k][mask] for k,mask in changed.items()])
report={'initialization_device_difference':{k:int(mask.sum()) for k,mask in changed.items()},
        'total_numel':veca.numel(),'difference_numel':subseta.numel(),'checkpoints':{}}
for path in sorted((ROOT/'URS/pretrained').glob('*.pt')):
    with torch.serialization.safe_globals([set]):
        ckpt=torch.load(path,map_location='cpu',weights_only=True)
    trained=ckpt['model_state_dict']
    assert list(trained)==names==keys
    vec=torch.cat([trained[k].flatten() for k in keys])
    delta=vec-veca
    sub=torch.cat([trained[k][mask] for k,mask in changed.items()])
    info={
        'delta_l2_over_init_l2':float(delta.norm()/veca.norm()),
        'delta_abs_quantiles':dict(zip(['50','90','95','99','100'],torch.quantile(delta.abs(),torch.tensor([.5,.9,.95,.99,1.])).tolist())),
        'delta_fraction_abs_gt_0_002':float((delta.abs()>.002).float().mean()),
        'ambiguous_coordinates_cosine_4090':float(torch.nn.functional.cosine_similarity(sub,subseta,dim=0)),
        'ambiguous_coordinates_cosine_3090':float(torch.nn.functional.cosine_similarity(sub,subsetb,dim=0)),
        'ambiguous_coordinates_l2_to_4090':float((sub-subseta).norm()),
        'ambiguous_coordinates_l2_to_3090':float((sub-subsetb).norm()),
    }
    info['candidate_initialization_group'] = '4090' if info['ambiguous_coordinates_cosine_4090'] > info['ambiguous_coordinates_cosine_3090'] else '3090'
    origin = veca if info['candidate_initialization_group'] == '4090' else vecb
    info['delta_l2_over_matching_candidate_init_l2'] = float((vec-origin).norm()/origin.norm())
    info['per_differing_tensor'] = {k: {
        'cosine_4090':float(torch.nn.functional.cosine_similarity(trained[k][mask],a[k][mask],dim=0)),
        'cosine_3090':float(torch.nn.functional.cosine_similarity(trained[k][mask],b[k][mask],dim=0)),
    } for k,mask in changed.items()}
    report['checkpoints'][path.stem]=info
    print(path.stem, 'delta/init',round(info['delta_l2_over_init_l2'],4),
          'ambiguous cosine4090/3090',round(info['ambiguous_coordinates_cosine_4090'],5),round(info['ambiguous_coordinates_cosine_3090'],5),
          'q95/q99',round(info['delta_abs_quantiles']['95'],5),round(info['delta_abs_quantiles']['99'],5),flush=True)
(RUN/'candidate_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
