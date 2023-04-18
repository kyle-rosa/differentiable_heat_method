import os
from pathlib import Path

import plotly.graph_objects as go
import torch
from torch import Tensor


def open_obj_file(
    fp: (str | os.PathLike),
    device=torch.device('cpu'),
    dtype=torch.float
) -> tuple[Tensor, Tensor]:
    """
        Opens simple WaveFront .obj files. 
        Extracts vertex coordinates and face data, but ignores textures.
        Assumes fairly strict formatting:
        - No line delimeters except \ n (without space).
        - All vertices have same number of components (assumed 3).
    """
    print(f"Opening {(fp.split('/')[-1])}... ", end='')
    verts, faces = [], []
    with open(fp) as file:
        for line in file:
            match line.split():
                case ['v', *coords]:
                    verts.append([float(it) for it in coords])
                case ['f', *idx]:
                    idx = [int(it.split('/')[0]) for it in idx]
                    for i in range(2, len(idx)):
                        faces.append([idx[0], idx[i-1], idx[i]])
    verts = torch.tensor(verts, device=device, dtype=dtype)
    faces = torch.tensor(faces, device=device, dtype=torch.long).subtract(1)
    print(f'found {len(verts)} vertices and {len(faces)} faces.')
    return (verts, faces)


def save_distance_viewer(
    mesh_name: str,
    source_vert: int,
    verts: Tensor,
    faces: Tensor,
    geodesic_distance: Tensor
) -> None:
    verts_array = verts.detach().cpu().numpy()
    faces_array = faces.detach().cpu().numpy()
    vert_features = geodesic_distance.sub(geodesic_distance.min(dim=0).values)
    vert_features = vert_features.div(vert_features.max(dim=0).values)
    vert_features_array = vert_features[..., 0].detach().cpu().numpy()
    go.Figure(
        data=[
            go.Mesh3d(
                x=verts_array[:, 0], y=-verts_array[:, 2], z=verts_array[:, 1],
                intensity=vert_features_array,
                i=faces_array[:, 0], j=faces_array[:, 1], k=faces_array[:, 2],
                showscale=True
            )
        ]
    ).update_layout(
        scene=dict(xaxis_title='x', yaxis_title='-z', zaxis_title='y'),
    ).write_html(Path() / 'output' / 'distance_field' / f'{mesh_name}-vertex_{source_vert}.html')


def save_gradient_viewer(
    mesh_name: str,
    source_vert: int,
    verts: Tensor,
    faces: Tensor,
    verts_cmap: Tensor
) -> None:
    verts_array = verts.detach().cpu().numpy()
    faces_array = faces.detach().cpu().numpy()
    verts_cmap = torch.nn.functional.normalize(verts_cmap, p=2, dim=-1)
    verts_cmap = verts_cmap.add(1).div(2)
    verts_cmap_array = verts_cmap.detach().cpu().numpy()
    go.Figure(
        data=[
            go.Mesh3d(
                x=verts_array[:, 0], y=-verts_array[:, 2], z=verts_array[:, 1],
                vertexcolor=verts_cmap_array,
                i=faces_array[:, 0], j=faces_array[:, 1], k=faces_array[:, 2],
                showscale=True
            )
        ]
    ).update_layout(
        scene=dict(xaxis_title='x', yaxis_title='-z', zaxis_title='y'),
    ).write_html(Path() / 'output' / 'gradient_field' / f'{mesh_name}-vertex_{source_vert}.html')
