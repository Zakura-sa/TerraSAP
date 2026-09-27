"""Resolve an explicit checkpoint or preserve legacy timm configuration behavior."""
import warnings


def pretrained_model_name(args, legacy_name='vit_base_patch16_224'):
    if 'pretrained_model_name' not in args:
        warnings.warn(
            f'Legacy config: using timm default weights for {legacy_name}. '
            'These are not verified paper weights; set pretrained_model_name explicitly.',
            UserWarning,
            stacklevel=2,
        )
        return legacy_name
    name = args['pretrained_model_name']
    if not isinstance(name, str) or not name.strip():
        raise ValueError('Set pretrained_model_name to a verified timm ViT-B/16 checkpoint identifier')
    return name.strip()
