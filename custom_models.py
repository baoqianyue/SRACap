# coding=utf-8
import os
import math
import copy
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CAP_TTA(nn.Module):
    def __init__(self, cap_model, device, momentum_update=False, update_freq=256, update_w=1.0, momentum=0.9998,
                 cap_ckpt=None):
       
        super().__init__()
        self.cap_model = cap_model
        self.cap_model.load_state_dict(torch.load(cap_ckpt, map_location=device))
        self.cap_model.to(device)

        self.device = device
        self.text_features = None
        self.image_features = None

        self.momentum_update = momentum_update
        self.update_freq = update_freq
        self.update_w = update_w
        self.momentum = momentum
        self.update_counter = 0
        with torch.no_grad():
            self.cap_state_dict = copy.deepcopy(self.cap_model.mapping_network.state_dict())
            self.initial_state_dict = copy.deepcopy(self.cap_model.mapping_network.state_dict())
            if self.momentum_update:
                self.momentum_state_dict = copy.deepcopy(self.cap_model.mapping_network.state_dict())


    @torch.no_grad()
    def reset_all(self):
        self.cap_model.mapping_network.load_state_dict(self.cap_state_dict)
        self.initial_state_dict = copy.deepcopy(self.cap_model.mapping_network.state_dict())
        if self.momentum_update:
            self.momentum_state_dict = copy.deepcopy(self.cap_model.mapping_network.state_dict())

    @torch.no_grad()
    def reset_initial(self):
        self.cap_model.mapping_network.load_state_dict(self.initial_state_dict)

    @torch.no_grad()
    def momentum_update_model(self):
        update_w = self.update_w
        if self.momentum_update:
            self.update_counter += 1
            state_dict = self.cap_model.mapping_network.state_dict()
            for k, v in state_dict.items():
                self.momentum_state_dict[k] = self.momentum * self.momentum_state_dict[k] + (1.0 - self.momentum) * v

            if self.update_counter >= self.update_freq:
                self.update_counter = 0
                for k, v in state_dict.items():
                    self.initial_state_dict[k] = (1 - update_w) * self.cap_state_dict[k] + update_w * \
                                                 self.momentum_state_dict[k]

    def parameters(self, recurse: bool = True):
        return self.cap_model.mapping_network.parameters()

    def train(self, mode: bool = True):
        return self.cap_model.train()

    @property
    def clip_project(self):
        return self.cap_model.mapping_network


    def forward(self, continuous_prompt: torch.Tensor, caption_tokens: torch.Tensor,
                hard_prompts_length: Optional[List] = None, mask: Optional[torch.Tensor] = None):
        return self.cap_model(continuous_prompt, caption_tokens, hard_prompts_length=hard_prompts_length, mask=mask)
