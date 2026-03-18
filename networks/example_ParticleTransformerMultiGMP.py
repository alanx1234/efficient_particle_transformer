import ast
import torch
from networks.parT import ParticleTransformer
from networks.logger import _logger

'''
Particle Transformer with multi-scale GMP and max_delta_r grid cap.

Each grid size covers the same physical region (ΔR < max_delta_r) at
a different resolution. Scale outputs are concatenated and projected
back to the original channel dim.

CLI usage example:
  --network-config networks/example_ParticleTransformerGMPMultiScale.py
  --network-kwargs gmp_grids="[0.025,0.05,0.1,0.2]" gmp_reduce=sum gmp_max_delta_r=0.8
'''


class ParticleTransformerWrapper(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.mod = ParticleTransformer(**kwargs)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mod.cls_token', }

    def forward(self, points, features, lorentz_vectors, mask):
        return self.mod(features, v=lorentz_vectors, mask=mask)


def get_model(data_config, **kwargs):
    # parse gmp_grids if passed as a string from CLI e.g. "[0.025,0.05,0.1,0.2]"
    if 'gmp_grids' in kwargs and isinstance(kwargs['gmp_grids'], str):
        kwargs['gmp_grids'] = ast.literal_eval(kwargs['gmp_grids'])

    cfg = dict(
        input_dim=len(data_config.input_dicts['pf_features']),
        num_classes=len(data_config.label_value),
        # network configurations
        pair_input_dim=4,
        use_pre_activation_pair=False,
        embed_dims=[128, 512, 128],
        pair_embed_dims=[64, 64, 64],
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        block_params=None,
        cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
        fc_params=[],
        activation='gelu',
        # misc
        trim=True,
        for_inference=False,
        # gmp
        use_gmp=True,
        gmp_coords='raw',
        gmp_kernel=3,
        gmp_grids=[0.025, 0.05, 0.1, 0.2],
        gmp_reduce='sum',
        gmp_max_delta_r=0.8,
    )
    cfg.update(**kwargs)
    cfg['use_gmp'] = True
    _logger.info('Model config: %s' % str(cfg))

    model = ParticleTransformerWrapper(**cfg)

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {
            **{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names},
            **{'softmax': {0: 'N'}},
        },
    }

    return model, model_info


def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()