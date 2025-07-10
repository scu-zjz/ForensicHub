import sys
import torch
from PIL import Image
from torchvision import transforms
from ForensicHub.registry import DATASETS, MODELS, POSTFUNCS, TRANSFORMS, build_from_registry


def load_image(image_path, image_size, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # (1, 3, H, W)
    return image


if __name__ == '__main__':
    # Set model configs here
    model_args = {
        "name": "ConvNextSmall",
        # model init config
        "init_config": {
            "image_size": 256
        },
        # model pretrained weight path
        "init_path": "/mnt/data1/dubo/workspace/ForensicHub/log/crossdataset_image_new/crossdataset_convnextsmall_none_train/checkpoint-19.pth"
    }

    model = build_from_registry(MODELS, model_args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load weights
    checkpoint = torch.load(model_args['init_path'], map_location=device)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)

    model.eval()

    # Load image
    image_path = '/mnt/data3/public_datasets/AIGC/GenImage/ADM/imagenet_ai_0508_adm/train/ai/375_adm_64.PNG'
    image_size = 256
    image_tensor = load_image(image_path=image_path, image_size=image_size).to(device)

    input_dict = {
        'image': image_tensor,
        # pseudo input
        "label": torch.ones(1).to(device),
        "mask": torch.ones(1, 1, image_size, image_size).to(device),
    }
    out_dict = model(**input_dict)
    pred_label = out_dict['pred_label'].item()

    print(f"Predicted label: {pred_label}")
