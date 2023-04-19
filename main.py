import glob
from pathlib import Path

import torch

from meshio import open_obj_file, save_distance_viewer, save_gradient_viewer
from operators import make_geodesic_distances

if __name__=='__main__':
    device = torch.device('cuda')
    dtype = torch.float64

    mesh_names = sorted([it.split('/')[-1][:-4] for it in glob.glob(str(Path() / 'meshes/*.obj'))])
    # mesh_names = ['bob_tri' for _ in range(5)]
    source_verts = [0,] * len(mesh_names)

    for (mesh_name, source_vert) in zip(mesh_names, source_verts):
        input_filepath = f"meshes/{mesh_name}.obj"

        (verts, faces_extrinsic) = open_obj_file(input_filepath, device=device, dtype=dtype)
        num_verts = verts.size(0)
        verts_features = torch.zeros((num_verts, 1), device=device, dtype=dtype)
        source_vert = torch.randint(0, len(verts), (1,)).item()
        verts_features[source_vert] = 1.0

        verts = torch.nn.Parameter(verts)
        vertex_areas, geodesic_distance = make_geodesic_distances(verts, faces_extrinsic, verts_features)
        (geodesic_distance * vertex_areas).sum().backward()

        save_gradient_viewer(mesh_name, source_vert, verts, faces_extrinsic, verts.grad)
        save_distance_viewer(mesh_name, source_vert, verts, faces_extrinsic, geodesic_distance)
