import torch
import trimesh
import numpy as np
from sdf import SDFLoss

if __name__ == "__main__":

    device = torch.device("cuda")

    faces = np.load("./example/data/faces.npy")
    verts = np.load("./example/data/verts.npy")
    with open("./example/data/labels.txt", "r") as f:
        skinning_label = f.readlines()
        skinning_label = [name.split(':')[-1].strip() for name in skinning_label]
    skinning_weights = np.load("./example/data/weights.npy")
    
    body_label = [joint for joint in skinning_label if "Spine" in joint or "Hips" in joint or "Shoulder" in joint]
    right_hand_label = [joint for joint in skinning_label if 'RightHand' in joint]

    skinning_group = {}
    vert2joint = np.argmax(skinning_weights, axis=-1)
    for index, label in enumerate(skinning_label):
        skinning_group[label] = (vert2joint==index).nonzero()[0]
    
    body_index = np.concatenate([skinning_group[label] for label in body_label])  
    right_hand_index = np.concatenate([skinning_group[label] for label in right_hand_label])

    points = torch.from_numpy(verts[body_index, :]).float().to(device)
    queries = torch.from_numpy(verts[right_hand_index, :]).float().to(device)
    queries[0, :] = points.mean(0)
    loss_fn = SDFLoss()
    loss = loss_fn(points, queries)

    # mesh = trimesh.points.PointCloud(vertices=verts[body_index, :])
    # mesh = mesh.convex_hull

    # vertices = torch.tensor(mesh.vertices).float().to(device)
    # faces = torch.tensor(mesh.faces).unsqueeze(0).to(device)

    # vmin = vertices.min(dim=0)[0]
    # vmax = vertices.max(dim=0)[0]
    # center = (vmax + vmin) / 2.0
    # scale = torch.max(vmax - vmin) / 2.0
    # # scale = 1.0

    # vertices = (vertices - center) / scale
    # vertices = vertices.unsqueeze(0)

    # scene = trimesh.Scene()
    # mesh = trimesh.Trimesh(vertices=vertices.cpu().squeeze(0).numpy(), faces=faces.cpu().squeeze(0).numpy(), face_colors = np.ones((faces.shape[1], 4)) * 100.)
    # # mesh.vertices = vertices.squeeze().cpu().numpy()
    # p = trimesh.points.PointCloud(vertices=vertices.squeeze().cpu().numpy())
    # scene.add_geometry(mesh)
    # scene.add_geometry(p)

    # start = [1.0, 1.0, 1.0]

    # line = trimesh.load_path(np.array([start,[0,0,0]]))
    # scene.add_geometry(line)
    # # scene.show()

    # query = torch.tensor(start).view(1,1,3).to(device)
    # vertices = torch.tensor([[[0, 0, 1],[0, 1, 0], [1, 0, 0]]]).float().to(device)
    # faces = torch.tensor([[[0, 1, 2]]]).long().to(device)
   
    # scene.add_geometry(mesh)

    # from sdf import SDF
    # sdf = SDF()(faces, vertices, query).cpu().numpy()
    # print(sdf)

    # line = trimesh.load_path(np.array([start,[sdf[0,0,0], sdf[0,0,1], sdf[0,0,2]]]))
    # scene.add_geometry(line)
    # scene.show()
    # sdf = SDF()