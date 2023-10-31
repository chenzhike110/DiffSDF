import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import sdf.csrc as _C

class SDFFunction(Function):
    """
    Definition of SDF function
    """

    @staticmethod
    def forward(ctx, phi, faces, vertices, query):
        return _C.sdf(phi, faces, vertices, query)

    @staticmethod
    def backward(ctx):
        return None, None, None

class SDF(nn.Module):

    def forward(self, faces, vertices, query):
        """
        phi: Batch x nV x 4 (nearest points + inside or outside)
        """
        phi = torch.zeros((query.shape[0], query.shape[1], query.shape[2]+1), device=vertices.device).float()
        phi = SDFFunction.apply(phi, faces, vertices, query)
        return phi