import glob
from pathlib import Path

import torch

from meshio import open_obj_file, save_distance_viewer, save_gradient_viewer
from operators import make_geodesic_distances

torch.set_float32_matmul_precision('high')


if __name__=='__main__':
    device = torch.device('cuda')
    dtype = torch.float64

    obj_filepaths = glob.glob(str(Path() / 'meshes/*.obj'))
    mesh_names = sorted([it.split('/')[-1][:-4] for it in obj_filepaths])
    source_verts = [0,] * len(mesh_names)

    for (mesh_name, src_vert) in zip(mesh_names, source_verts):
        input_filepath = f"meshes/{mesh_name}.obj"
        (verts, faces_extrinsic) = open_obj_file(input_filepath, device, dtype)

        verts_feats = torch.zeros((verts.size(0), 1), device=device, dtype=dtype)
        verts_feats[src_vert] = 1.0

        verts = torch.nn.Parameter(verts)
        (vert_areas, geodesic_dist) = make_geodesic_distances(
            verts, faces_extrinsic, verts_feats
        )
        (geodesic_dist * vert_areas).sum().backward()

        save_gradient_viewer(mesh_name, src_vert, verts, faces_extrinsic, verts.grad)
        save_distance_viewer(mesh_name, src_vert, verts, faces_extrinsic, geodesic_dist)
