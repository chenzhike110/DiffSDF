from sdf import *

if __name__ == "__main__":
    verts = torch.tensor([[-2, 0, 0], [0.1, 0.1, 0.1]]).float()
    mesh_points = torch.tensor([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
        [0, 0, -1]
    ]).float()
    mesh = trimesh.points.PointCloud(vertices=mesh_points.squeeze().numpy())
    mesh = mesh.convex_hull
    normals = mesh.face_normals
    faces = torch.tensor(mesh.faces)
    faces = mesh_points[faces]
    d = SDF()(verts, mesh_points)
    dist, point = SDF.point_triangle_dist(verts, faces, normals)

    point = point.reshape(-1, 3)
    print(dist)

    # verts = torch.cat((verts, *pro), 0)
    start = verts.repeat(faces.shape[0], 1).reshape(-1, 3)

    lines = np.concatenate((start, point), axis=1).reshape(point.shape[0],2,3)
    lines = trimesh.load_path(lines)

    faces_center = faces.mean(dim=1).numpy()

    lines_faces = np.concatenate((faces_center, faces_center+normals), axis=1).reshape(faces_center.shape[0],2,3)
    lines_faces = trimesh.load_path(lines_faces)

    scene = trimesh.Scene()
    p = trimesh.points.PointCloud(vertices=verts.numpy())
    scene.add_geometry(p)
    scene.add_geometry(mesh)
    scene.add_geometry(lines)
    # scene.add_geometry(lines_faces)
    scene.show(flags={'wireframe': True, 'axis': True})