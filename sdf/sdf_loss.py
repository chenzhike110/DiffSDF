import torch
import trimesh
import torch.nn as nn
import numpy as np

from sdf import SDF

class SDFLoss(nn.Module):

    def __init__(self):
        super(SDFLoss, self).__init__()
        self.sdf = SDF()

    def forward(self, points, queries):
        mesh = trimesh.points.PointCloud(vertices=points.cpu().numpy())
        mesh = mesh.convex_hull
        faces = torch.tensor(mesh.faces).to(points.device)
        verts = torch.tensor(mesh.vertices).float().to(points.device)

        vmin = verts.min(dim=0)[0]
        vmax = verts.max(dim=0)[0]
        center = (vmax + vmin) / 2.0
        scale = torch.max(vmax - vmin) / 2.0

        verts = (verts - center) / scale
        queries = (queries - center) / scale

        phi = self.sdf(faces, verts, queries)

        phi[:, :-1] = phi[:, :-1] *  scale + center
        queries = queries * scale + center
        
        distance = torch.norm(queries - phi[ :, :-1], dim=-1)

        #debug
        scene = trimesh.Scene()
        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, face_colors = np.ones((mesh.faces.shape[0], 4)) * 100.)
        p = trimesh.points.PointCloud(vertices=queries.cpu().numpy())
        lines_faces = np.concatenate((queries.cpu().numpy(), phi[:, :-1].cpu().numpy()), axis=1).reshape(queries.shape[0],2,3)
        lines_faces = trimesh.load_path(lines_faces)
        scene.add_geometry(mesh)
        scene.add_geometry(p)
        scene.add_geometry(lines_faces)
        scene.show()

        inside = (phi[:, -1] < 1).nonzero()
        distance[inside] *= -1
        return distance
