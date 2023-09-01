import torch
import trimesh
import numpy as np
from torch import nn

class SDF(nn.Module):
    """
    signed distance field with given points and mesh faces
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, p, v, f=None):
        """
        p: given query points Nx3
        v: given vertices of one mesh Nx3
        f: given faces of one mesh Nx3 (if None, a convex hull will be created with v)
        """
        f_normals = None

        if f is None:
            mesh = trimesh.points.PointCloud(vertices=v.squeeze().numpy())
            mesh = mesh.convex_hull
            f_normals = mesh.face_normals
            faces = torch.tensor(mesh.faces).to(v.device)
            faces = v[faces]
        else:
            faces = v[f]
            f_normals = torch.cross(faces[:, 0]-faces[:, 1], faces[:, 2]-faces[:, 1], dim=1)

        dist, closest_p = self.point_triangle_dist(p, faces, f_normals)
        _, indices = torch.abs(dist).min(dim=0, keepdim=True)
        dist = torch.gather(dist, 0, indices)
        return dist

    @staticmethod
    def point_triangle_dist(p, faces, face_normals):
        """
        point to triangle distance function barycentric coordinates
        """
        e1 = faces[:, 0] - faces[:, 1]
        e2 = faces[:, 2] - faces[:, 1]
        e3 = faces[:, 2] - faces[:, 0]
        p0 = faces[:, 0].repeat(1, p.shape[0]).reshape(-1, 3)
        p1 = faces[:, 1].repeat(1, p.shape[0]).reshape(-1, 3)
        p2 = faces[:, 2].repeat(1, p.shape[0]).reshape(-1, 3)
        e1_extend = e1.repeat(1, p.shape[0]).reshape(-1, 3)
        e2_extend = e2.repeat(1, p.shape[0]).reshape(-1, 3)
        e3_extend = e3.repeat(1, p.shape[0]).reshape(-1, 3)
        
        face_norm = torch.tensor(face_normals).float().to(p.device)
        face_norm_extend = face_norm.repeat(1, p.shape[0]).reshape(-1, 3)

        p_extend = p.repeat(faces.shape[0], 1)
        
        tvec = p_extend - p2

        det = torch.matmul(face_norm_extend.unsqueeze(1), tvec.unsqueeze(-1)).squeeze(-1)
        outside = det > 0
        # print("outsidec: ", outside)
        projection = p_extend - (face_norm_extend * det.repeat(1, 3))

        # check point inside
        # ax1+bx2=x3 ay1+by2=y3 az1+bz2=z3
        pt0 = p0 - projection
        pt1 = p1 - projection
        pt2 = p2 - projection
        u = torch.cross(pt1, pt2, dim=1)
        v = torch.cross(pt2, pt0, dim=1)
        w = torch.cross(pt0, pt1, dim=1)
        dot_uv = torch.matmul(u.unsqueeze(1), v.unsqueeze(-1)).squeeze(-1)
        dot_uw = torch.matmul(u.unsqueeze(1), w.unsqueeze(-1)).squeeze(-1)
        # a = (pt1[:, 0] * e2_extend[:, 1] - pt1[:, 1] * e2_extend[:, 0]) / (e1_extend[:, 0] * e2_extend[:, 1] - e1_extend[:, 1] * e2_extend[:, 0])
        # b = (pt1[:, 0] * e1_extend[:, 1] - pt1[:, 1] * e1_extend[:, 0]) / (e2_extend[:, 0] * e1_extend[:, 1] - e2_extend[:, 1] * e1_extend[:, 0])
        dp = torch.norm(projection-p_extend, dim=-1)
        mask1 = torch.logical_and(dot_uv >= 0,  dot_uw >= 0).squeeze()
        
        # e1
        m2 = torch.norm(e1_extend, dim=-1)
        m2 = m2 * m2
        s12 = (torch.matmul(p0.unsqueeze(1), e1_extend.unsqueeze(-1)) - torch.matmul(p_extend.unsqueeze(1), e1_extend.unsqueeze(-1))).squeeze() / m2
        s12 = torch.clamp(s12, 0., 1.).unsqueeze(-1).repeat(1, 3)
        closest12 = s12 * p1 + (1 - s12) * p0
        d12 = torch.norm(closest12-p_extend, dim=-1)

        # e2
        m2 = torch.norm(e2_extend, dim=-1)
        m2 = m2 * m2
        s23 = (torch.matmul(p2.unsqueeze(1), e2_extend.unsqueeze(-1)) - torch.matmul(p_extend.unsqueeze(1), e2_extend.unsqueeze(-1))).squeeze() / m2
        s23 = torch.clamp(s23, 0., 1.).unsqueeze(-1).repeat(1, 3)
        closest23 = s23 * p1 + (1 - s23) * p2
        d23 = torch.norm(closest23-p_extend, dim=-1)

        # e3
        m2 = torch.norm(e3_extend, dim=-1)
        m2 = m2 * m2
        s31 = (torch.matmul(p2.unsqueeze(1), e3_extend.unsqueeze(-1)) - torch.matmul(p_extend.unsqueeze(1), e3_extend.unsqueeze(-1))).squeeze() / m2
        s31 = torch.clamp(s31, 0., 1.).unsqueeze(-1).repeat(1, 3)
        closest31 = s31 * p0 + (1 - s31) * p2
        d31 = torch.norm(closest31-p_extend, dim=-1)

        d_all = torch.stack([d12, d23, d31], dim=0)
        values, indices = d_all.min(dim=0, keepdim=True)
        indices = indices.unsqueeze(-1).repeat(1, 1, 3)

        closest = torch.stack([closest12, closest23, closest31], dim=0)
        closest = torch.gather(closest, 0, indices).view(-1, 3)

        result = values.squeeze()
        result[mask1] = dp[mask1]
        closest[mask1.unsqueeze(-1).repeat(1, 3)] = projection[mask1.unsqueeze(-1).repeat(1, 3)]
        # closest[mask1] = 

        outside = (outside.float() - 0.5) * 2.0
        result = result * outside.squeeze()
        result = result.reshape(faces.shape[0], p.shape[0])
        closest = closest.reshape(faces.shape[0], p.shape[0], 3)

        return result, closest