import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from ForensicHub.registry import register_model
from ForensicHub.core.base_model import BaseModel
import torchvision.models as models
import yaml


@register_model("Capsule_net")
class CapsuleNet(BaseModel):
    def __init__(self, yaml_config_path):
        super(CapsuleNet, self).__init__()

        # 从 YAML 配置文件读取配置
        with open(yaml_config_path, 'r') as file:
            config = yaml.safe_load(file)

        # 配置中的类数量
        self.num_classes = config['num_classes']

        # Initialize CapsuleNet components
        self.vgg_ext = VggExtractor()
        self.fea_ext = FeatureExtractor()
        self.fea_ext.apply(self.weights_init)

        # Routing layer
        self.routing_stats = RoutingLayer(
            num_input_capsules=10,
            num_output_capsules=self.num_classes,
            data_in=8,
            data_out=4,
            num_iterations=2
        )

        # Initialize CapsuleLoss directly
        self.loss_func = CapsuleLoss()  # Use CapsuleLoss directly

    def features(self, image: torch.Tensor) -> torch.tensor:
        """
        Extract features from input image using VGG extractor and feature extractor.

        Args:
            image (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Extracted feature tensor.
        """
        input = self.vgg_ext(image)  # Extract features using VGG extractor
        feature = self.fea_ext(input)  # Further process with feature extractor
        return feature

    def classifier(self, features: torch.tensor) -> torch.tensor:
        """
        Perform classification by applying routing statistics on the features.

        Args:
            features (torch.Tensor): Extracted features from the image.

        Returns:
            torch.Tensor: Class prediction probabilities and mean class.
        """
        z = self.routing_stats(features, random=False, dropout=0.0)
        classes = F.softmax(z, dim=-1)
        class_ = classes.detach()
        class_ = class_.mean(dim=1)
        return classes, class_

    def forward(self, image: torch.Tensor, label: torch.Tensor, **kwargs) -> dict:
        """
        Perform forward propagation.

        Args:
            image (torch.Tensor): Input image tensor.
            label (torch.Tensor): Ground truth label tensor.

        Returns:
            dict: Dictionary containing the backward loss and predictions.
        """
        label = label.long()
        # Extract features from the input image
        features = self.features(image)

        # Get the prediction by classifier
        preds, pred = self.classifier(features)

        # Get the probability of the pred
        prob = torch.softmax(pred, dim=1)[:, 1]

        # Compute the loss using the ground truth labels
        loss = self.loss_func(preds, label)

        # Return prediction and loss values
        return {
            "backward_loss": loss,
            "pred_label": prob,
            "visual_loss": {"combined_loss": loss}
        }

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


# VGG input(10,3,256,256)
class VggExtractor(nn.Module):
    def __init__(self, train=False):
        super(VggExtractor, self).__init__()
        self.vgg_1 = self.Vgg(models.vgg19(pretrained=True), 0, 18)
        if train:
            self.vgg_1.train(mode=True)
            self.freeze_gradient()
        else:
            self.vgg_1.eval()

    def Vgg(self, vgg, begin, end):
        features = nn.Sequential(*list(vgg.features.children())[begin:(end + 1)])
        return features

    def freeze_gradient(self, begin=0, end=9):
        for i in range(begin, end + 1):
            self.vgg_1[i].requires_grad = False

    def forward(self, input):
        return self.vgg_1(input)


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.NO_CAPS = 10
        self.capsules = nn.ModuleList([nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            StatsNet(),
            nn.Conv1d(2, 8, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(8),
            nn.Conv1d(8, 1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(1),
            View(-1, 8),
        ) for _ in range(self.NO_CAPS)])

    def squash(self, tensor, dim):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / (torch.sqrt(squared_norm))

    def forward(self, x):
        outputs = [capsule(x) for capsule in self.capsules]
        output = torch.stack(outputs, dim=-1)
        return self.squash(output, dim=-1)


class StatsNet(nn.Module):
    def __init__(self):
        super(StatsNet, self).__init__()

    def forward(self, x):
        x = x.view(x.data.shape[0], x.data.shape[1], x.data.shape[2] * x.data.shape[3])
        mean = torch.mean(x, 2)
        std = torch.std(x, 2)
        return torch.stack((mean, std), dim=1)


class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(self.shape)


# Capsule right Dynamic routing
class RoutingLayer(nn.Module):
    def __init__(self, num_input_capsules, num_output_capsules, data_in, data_out, num_iterations):
        super(RoutingLayer, self).__init__()

        self.num_iterations = num_iterations
        self.route_weights = nn.Parameter(torch.randn(num_output_capsules, num_input_capsules, data_out, data_in))

    def squash(self, tensor, dim):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / (torch.sqrt(squared_norm))

    def forward(self, x, random=False, dropout=0.0):
        x = x.transpose(2, 1)
        if random:
            noise = torch.Tensor(0.01 * torch.randn(*self.route_weights.size())).cuda()
            route_weights = self.route_weights + noise
        else:
            route_weights = self.route_weights
        priors = route_weights[:, None, :, :, :] @ x[None, :, :, :, None]
        priors = priors.transpose(1, 0)
        if dropout > 0.0:
            drop = torch.Tensor(torch.FloatTensor(*priors.size()).bernoulli(1.0 - dropout)).cuda()
            priors = priors * drop
        logits = torch.Tensor(torch.zeros(*priors.size())).to(priors.device)
        for i in range(self.num_iterations):
            probs = F.softmax(logits, dim=2)
            outputs = self.squash((probs * priors).sum(dim=2, keepdim=True), dim=3)
            if i != self.num_iterations - 1:
                delta_logits = priors * outputs
                logits = logits + delta_logits
        outputs = outputs.squeeze()
        if len(outputs.shape) == 3:
            outputs = outputs.transpose(2, 1).contiguous()
        else:
            outputs = outputs.unsqueeze_(dim=0).transpose(2, 1).contiguous()
        return outputs


# Loss Function: CapsuleLoss
class CapsuleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        """
        Computes the capsule loss.

        Args:
            inputs: A PyTorch tensor of size (batch_size, num_classes) containing the predicted scores.
            targets: A PyTorch tensor of size (batch_size) containing the ground-truth class indices.

        Returns:
            A scalar tensor representing the capsule loss.
        """
        loss_t = self.cross_entropy_loss(inputs[:, 0, :], targets)

        for i in range(inputs.size(1) - 1):
            loss_t = loss_t + self.cross_entropy_loss(inputs[:, i + 1, :], targets)
        return loss_t
