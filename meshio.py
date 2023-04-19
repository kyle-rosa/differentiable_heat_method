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
    Opens a WaveFront .obj file and extracts vertex coordinates and face data.
    
    Args:
    - fp (str | os.PathLike): The file path to the .obj file to open.
    - device (torch.device): The device to use for the returned tensors. Defaults to 'cpu'.
    - dtype (torch.dtype): The data type to use for the returned tensors. Defaults to torch.float.
    
    Returns:
    - tuple[Tensor, Tensor]: A tuple of two tensors containing the vertex coordinates and face data.
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
    """
    Saves a 3D mesh visualization to an HTML file, with vertex colors based on geodesic distances from a source vertex.
    
    Args:
    - mesh_name (str): A name for the mesh being visualized.
    - source_vert (int): The index of the source vertex to use for geodesic distance calculation.
    - verts (Tensor): A tensor of shape (V, 3) containing the 3D coordinates of the mesh's vertices.
    - faces (Tensor): A tensor of shape (F, 3) containing the indices of the vertices forming each face of the mesh.
    - geodesic_distance (Tensor): A tensor of shape (V,) containing the geodesic distance from the source vertex to each vertex.
    
    Returns:
    - None
    
    The function saves the visualization to an HTML file in the 'output/distance_field' directory.
    """
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
    """
    Saves a 3D mesh visualization to an HTML file, with vertex colors based on a color map specified for each vertex.
    
    Args:
    - mesh_name (str): A name for the mesh being visualized.
    - source_vert (int): The index of the vertex to use as the source for the color map.
    - verts (Tensor): A tensor of shape (N, 3) containing the 3D coordinates of the mesh's vertices.
    - faces (Tensor): A tensor of shape (F, 3) containing the indices of the vertices forming each face of the mesh.
    - verts_cmap (Tensor): A tensor of shape (N, C) containing the color map values for each vertex. C is the number of color channels.
    
    Returns:
    - None
    
    The function saves the visualization to an HTML file in the 'output/gradient_field' directory.
    """
    verts_array = verts.detach().cpu().numpy()
    faces_array = faces.detach().cpu().numpy()
    verts_cmap = torch.nn.functional.normalize(verts_cmap, p=2, dim=-1).add(1).div(2)
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
