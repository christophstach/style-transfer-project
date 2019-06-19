import torch
import torch.nn as nn
import torch.nn.functional as F

from csfnst.utils import extract_activation_maps, gram_matrix


class PerceptualLoss(nn.Module):
    def __init__(
            self,
            model,
            content_layers,
            style_layers,
            style_image,
            content_weight=1,
            style_weight=1e7,
            total_variation_weight=0,
    ):
        super(PerceptualLoss, self).__init__()

        self.model = model
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.total_variation_weight = total_variation_weight

        style_features = extract_activation_maps(
            style_image.unsqueeze(0),
            self.model,
            style_layers,
            detach=True
        )

        self.content_loss_val = None
        self.style_loss_val = None
        self.total_variation_loss_val = None
        self.loss_val = None

        self.style_grams = [gram_matrix(style_feature) for style_feature in style_features]

    def forward(self, generated_images, input_images):
        if generated_images.shape != input_images.shape:
            input_images = F.interpolate(input_images, size=(generated_images.shape[2], generated_images.shape[3]))

        self.content_loss_val = self.content_weight * self.content_loss(generated_images, input_images)
        self.style_loss_val = self.style_weight * self.style_loss(generated_images)
        self.total_variation_loss_val = self.total_variation_weight * self.total_variation_loss(generated_images)
        self.loss_val = self.content_loss_val + self.style_loss_val + self.total_variation_loss_val

        return self.loss_val

    def content_loss(self, generated_images, input_images):
        generated_features = extract_activation_maps(generated_images, self.model, self.content_layers)
        input_features = extract_activation_maps(input_images, self.model, self.content_layers, detach=True)

        return sum([
            F.mse_loss(
                generated_feature,
                input_feature
            )
            for generated_feature, input_feature in zip(generated_features, input_features)
        ])

    def style_loss(self, generated_images):
        generated_features = extract_activation_maps(generated_images, self.model, self.style_layers)
        generated_grams = [
            gram_matrix(generated_feature)
            for generated_feature in generated_features
        ]

        return sum([
            F.mse_loss(
                generated_gram,
                torch.cat([
                    style_gram
                    for _ in range(generated_gram.shape[0])
                ])
            )
            for generated_gram, style_gram in zip(generated_grams, self.style_grams)
        ])

    def total_variation_loss(self, generated_images):
        return torch.add(
            torch.sum(torch.abs(generated_images[:, :, :, :-1] - generated_images[:, :, :, 1:])),
            torch.sum(torch.abs(generated_images[:, :, :-1, :] - generated_images[:, :, 1:, :]))
        )
