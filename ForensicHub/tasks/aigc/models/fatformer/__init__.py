from .clip_models import CLIPModel

VALID_NAMES = [
    'CLIP:ViT-B/32', 
    'CLIP:ViT-B/16', 
    'CLIP:ViT-L/14', 
]

def build_model(args):
    if args.backbone.startswith("CLIP:"):
        assert args.backbone in VALID_NAMES
        return CLIPModel(args.backbone[5:], args)
    else:
        raise NotImplementedError