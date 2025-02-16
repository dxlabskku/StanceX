import sys
sys.path.append('./')
from collections import OrderedDict
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer
from models.stanceX.stanceX_textual_model import stanceXTextualModel
from models.stanceX.stanceX_visual_model import stanceXVisualModel

import torch.nn.functional as F
import numpy as np
import os
import logging

class stanceXModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        num_layers_to_freeze = args.frozen_layers

        textual_config = AutoConfig.from_pretrained(args.textual_transformer_name)
        visual_config = AutoConfig.from_pretrained(args.visual_transformer_name)

        self.textual_hidden_size = textual_config.hidden_size
        self.visual_hidden_size = visual_config.hidden_size

        self.stanceX_textual_model = stanceXTextualModel(args, num_layers_to_freeze)
        self.stanceX_visual_model = stanceXVisualModel(args, num_layers_to_freeze)

        if args.linear_injection == -1:
            linear_injection = min(self.textual_hidden_size, self.visual_hidden_size)
        else:
            linear_injection = args.linear_injection

        self.textual_transformer_linear = nn.Sequential(
            nn.Linear(self.textual_hidden_size, linear_injection),
            nn.LayerNorm([linear_injection]),
            nn.LeakyReLU(0.2)
        )
        self.visual_transformer_linear = nn.Sequential(
            nn.Linear(self.visual_hidden_size, linear_injection),
            nn.LayerNorm([linear_injection]),
            nn.LeakyReLU(0.2)
        )
        self.classifier = nn.Linear(linear_injection * 2, args.label_size)

    def forward(self, input_data):
        tweet_outputs = self.stanceX_textual_model(input_data)
        image_outputs = self.stanceX_visual_model(input_data)

        textual_transformed = tweet_outputs['last_hidden_state']
        visual_transformed = image_outputs['last_hidden_state']

        textual_features = self.textual_transformer_linear(textual_transformed.mean(dim=1))
        visual_transformed = visual_transformed.unsqueeze(1)

        visual_features = self.visual_transformer_linear(visual_transformed.mean(dim=1))

        combined_features = torch.cat([textual_features, visual_features], dim=-1)

        logits = self.classifier(combined_features)
        return logits

if __name__ == '__main__':
    class Args():
        label_size = 3
        linear_injection = -1
        frozen_layers = 6
        textual_transformer_name = 'vinai/bertweet-base'
        visual_transformer_name = 'google/vit-base-patch16-224'

    args = Args()
    model = stanceXModel(args)
    text_ids = torch.randint(low=0, high=64001, size=[16, 128], dtype=torch.long)
    text_masks = torch.ones(size=[16, 128], dtype=torch.float32)
    image_tensor = torch.randn(size=[16, 3, 224, 224])
    input_data = {'input_ids': text_ids, 'attention_mask': text_masks, 'pixel_values': image_tensor}

    logits = model(input_data)
