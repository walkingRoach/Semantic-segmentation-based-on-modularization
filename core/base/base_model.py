import logging
import torch.nn as nn
import numpy as np


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self, x):
        raise NotImplementedError

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])

        self.logger.info('Number of trainable parameters: {}'.format(n_params))

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])

        return super(BaseModel, self).__str__() + '\nNumber of trainable parameters : {}'.format(n_params)

    def get_backbone_params(self):
        raise NotImplementedError

    def get_decoder_params(self):
        raise NotImplementedError
