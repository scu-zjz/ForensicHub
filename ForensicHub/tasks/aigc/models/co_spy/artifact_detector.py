import torch
from diffusers import StableDiffusionPipeline
from .artifact_extractor import VAEReconEncoder
from torchvision import transforms
from .cospy_utils import data_augment


# Artifact Detector (Extract artifact features using VAE)
class ArtifactDetector(torch.nn.Module):
    def __init__(self, dim_artifact=512, num_classes=1):
        super(ArtifactDetector, self).__init__()
        # Load the pre-trained VAE
        model_id = "CompVis/stable-diffusion-v1-4"
        vae = StableDiffusionPipeline.from_pretrained(model_id).vae
        # Freeze the VAE visual encoder
        vae.requires_grad_(False)
        self.artifact_encoder = VAEReconEncoder(vae)

        # Classifier
        self.fc = torch.nn.Linear(dim_artifact, num_classes)

        # Normalization
        self.mean = [0.0, 0.0, 0.0]
        self.std = [1.0, 1.0, 1.0]

        # Resolution
        self.loadSize = 224
        self.cropSize = 224

        # Data augmentation
        self.blur_prob = 0.0
        self.blur_sig = [0.0, 3.0]
        self.jpg_prob = 0.5
        self.jpg_method = ['cv2', 'pil']
        self.jpg_qual = list(range(70, 96))

        # Define the augmentation configuration
        self.aug_config = {
            "blur_prob": self.blur_prob,
            "blur_sig": self.blur_sig,
            "jpg_prob": self.jpg_prob,
            "jpg_method": self.jpg_method,
            "jpg_qual": self.jpg_qual,
        }

        # Pre-processing
        crop_func = transforms.RandomCrop(self.cropSize)
        flip_func = transforms.RandomHorizontalFlip()
        rz_func = transforms.Resize(self.loadSize)
        aug_func = transforms.Lambda(lambda x: data_augment(x, self.aug_config))

        self.train_transform = transforms.Compose([
            aug_func,
            rz_func,
            crop_func,
            flip_func,
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

        self.test_transform = transforms.Compose([
            rz_func,
            crop_func,
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def forward(self, x, return_feat=False):
        feat = self.artifact_encoder(x)
        out = self.fc(feat)
        if return_feat:
            return feat, out
        return out

    def save_weights(self, weights_path):
        save_params = {k: v.cpu() for k, v in self.state_dict().items()}
        torch.save(save_params, weights_path)

    def load_weights(self, weights_path):
        weights = torch.load(weights_path)
        self.load_state_dict(weights)
