from pathlib import Path
import glob
import torch
import torch.nn.functional as F
from operators import make_geodesic_distances
from meshio import open_obj_file, save_distance_viewer, save_gradient_viewer


if __name__=='__main__':
    device = torch.device('cuda')
    dtype = torch.float64

    mesh_names = sorted([it.split('/')[-1][:-4] for it in glob.glob(str(Path() / 'meshes/*.obj'))])
    source_verts = [0,] * len(mesh_names)

    for (mesh_name, source_vert) in zip(mesh_names, source_verts):
        input_filepath = f"meshes/{mesh_name}.obj"

        (verts, faces_extrinsic) = open_obj_file(input_filepath, device=device, dtype=dtype)
        verts_features = torch.zeros((verts.shape[0], 1), device=device, dtype=dtype)
        verts_features[source_verts] = 1

        verts = torch.nn.Parameter(verts)
        geodesic_distance = make_geodesic_distances(verts, faces_extrinsic, verts_features)
        geodesic_distance.sum().backward()

        save_gradient_viewer(mesh_name, source_vert, verts, faces_extrinsic, verts.grad)
        save_distance_viewer(mesh_name, source_vert, verts, faces_extrinsic, geodesic_distance)
