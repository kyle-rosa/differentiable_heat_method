import torch
from itertools import product


def cross_product(U, V):
    return (
        U[..., [1, 2, 0]].mul(V[..., [2, 0, 1]])
        .sub(U[..., [2, 0, 1]].mul(V[..., [1, 2, 0]]))
    )


def make_mesh_geometry(edges):
    """
    Given the tangent half-edge vectors with shape (F, 3, 3),
    returns:
    - the squared length of each edge, shape (F, 3),
    - the area of intersections between faces and vertex dual cells, shape (F, 3),
    - the cotangents of the angles at each corner of each face, shape (F, 3)
    """
    edge_lengths2 = edges.pow(2).sum(dim=-1)
    dots = -edges[..., [1, 2, 0], :].multiply(edges[..., [2, 0, 1], :]).sum(dim=-1)
    crosses = cross_product(-edges[..., [1, 2, 0], :], edges[..., [2, 0, 1], :]).norm(dim=-1)
    cots = dots.div(crosses)
    voronoi_areas = edge_lengths2.multiply(cots)[..., [[1, 2], [2, 0], [0, 1]]].sum(dim=-1).div(8)
    alternative_areas = voronoi_areas.sum(dim=-1, keepdim=True).multiply(cots.lt(0).add(1).div(4))
    cell_areas = voronoi_areas.where(cots.gt(0).all(dim=-1, keepdim=True), alternative_areas)
    return (edge_lengths2, cell_areas, cots)


def make_face_gradients(d_verts, features, face_areas, faces):
    """
        d_verts: (F, 3, 3) float
        features: (V, C) float
        face_gradients: (F, C, 3) float
    """
    normals = cross_product(d_verts[..., [1, 2, 0], :], d_verts[..., [2, 0, 1], :])
    tangents = cross_product(normals, d_verts)
    return (
        tangents[..., None, :]
        .multiply(features[faces, :, None])
        .sum(dim=-3)
        .div(face_areas.multiply(2).pow(2)[:, None, None])
    )


def differential(verts_features, faces):
    faces_verts = verts_features[faces]  # (F, 3, 3)
    d_verts = faces_verts[..., [2, 0, 1], :].sub(faces_verts[..., [1, 2, 0], :])  # (F, 3, 3)
    return d_verts


def sum_cell_values_(cell_values, faces, out):
    """
        Aggregate the (F, 3) cell_values to the vertices, output has shape (V,).
    """
    return out.mul(0.).index_add_(
        dim=0, 
        index=faces.reshape(-1), 
        source=cell_values.reshape(-1)
    )

def sum_cell_values(cell_values, faces):
    """
        Aggregate the (F, 3) cell_values to the vertices, output has shape (V,).
    """
    V = faces.max().add(1).item()  # C = cell_values.shape[-1]
    return torch.zeros(
        (V, ),
        device=cell_values.device,
        dtype=cell_values.dtype,
    ).index_add_(
        dim=0, 
        index=faces.reshape(-1), 
        source=cell_values.reshape(-1)
    )


def make_local_cotan_weights(device="cpu", dtype=torch.float):
    local_cotan_kernel = torch.zeros((3, 3, 3), device=device, dtype=dtype)
    local_cotan_kernel[[0, 1, 2], [0, 1, 2], [1, 2, 0]] = 1
    local_cotan_kernel[[0, 1, 2], [0, 1, 2], [2, 0, 1]] = 1
    local_cotan_kernel[[0, 1, 2], [1, 2, 0], [2, 0, 1]] = -1
    local_cotan_kernel[[0, 1, 2], [2, 0, 1], [1, 2, 0]] = -1
    return local_cotan_kernel



def make_local_indices(device="cpu"):
    return torch.tensor(
        list(product(*[range(3)] * 2)), device=device, dtype=torch.long
    ).view(3, 3, 2)


_local_indices = make_local_indices(device=torch.device('cuda'))
_local_cotan_weights = make_local_cotan_weights(device=torch.device('cuda'), dtype=torch.float64)


def make_conformal_laplacian_kernel(cots, faces, local_cotan_weights=_local_cotan_weights):
    V = faces.max().add(1).item()
    x = faces[:, _local_indices]
    indexadd_laplacian_kernel = (
        torch.zeros((V, V), device=cots.device, dtype=cots.dtype)
        .reshape(-1)
        .index_add_(
            dim=0,
            index=(x[..., 0] * V + x[..., 1]).reshape(-1),
            source=torch.einsum("ijc, fc -> fij", local_cotan_weights, cots).reshape(-1),
        )
        .reshape(V, V)
        .div(2)
    )
    return indexadd_laplacian_kernel


def make_conformal_laplacian(features, cots, faces, local_cotan_weights):
    index = faces[:, _local_indices].reshape(-1, 2).T
    values = torch.einsum("ijc, fc -> fij", local_cotan_weights, cots).reshape(-1, 1)
    conformal_laplacian = (
        torch.zeros_like(features)
        .index_add_(
            dim=0, 
            index=index[0, :], 
            source=features[index[1, :], :].multiply(values)
        ).div(2)
    )
    return conformal_laplacian


def make_laplacian(conformal_laplacian, vertex_areas):
    return conformal_laplacian.div(vertex_areas[..., None])


def diffuse_features_backward_euler(vertex_areas, conformal_laplacian_kernel, verts_features, tau):
    return torch.linalg.solve(
        conformal_laplacian_kernel.multiply(tau).add(vertex_areas.diag()),
        verts_features.multiply(vertex_areas[..., None]),
    )

def make_integrated_divergence(d_verts, face_gradients, cots, faces):
    V = faces.max().add(1).item()
    C = face_gradients.shape[-2]
    idxs = faces.view(-1)[None]
    vals = (
        d_verts[..., None, :]
        .multiply(face_gradients[..., None, :, :])
        .sum(dim=-1)
        .multiply(cots[..., None])[..., [[1, 2], [2, 0], [0, 1]], :]
        .diff(n=1, dim=-2)[..., 0, :]
        .div(2)
    ).view(-1, C)
    buffer = (
        torch.zeros((V, C), device=cots.device, dtype=cots.dtype)
        .index_add_(
            dim=0,
            index=idxs[0],
            source=vals,
        )
    )
    return buffer 


def make_geodesic_distances(verts, faces_intrinsic, verts_features):
    d_verts_intrinsic = differential(verts, faces_intrinsic)
    (edge_lengths2_intrinsic, cell_areas, cots) = make_mesh_geometry(d_verts_intrinsic)
    face_areas = cell_areas.sum(dim=-1)
    vertex_areas = sum_cell_values(cell_areas, faces_intrinsic)
    conformal_laplacian_kernel = make_conformal_laplacian_kernel(cots, faces_intrinsic)
    tau = edge_lengths2_intrinsic.mean()
    diffused_verts_features = diffuse_features_backward_euler(vertex_areas, conformal_laplacian_kernel, verts_features, tau)
    diffused_verts_features_grad = make_face_gradients(d_verts_intrinsic, diffused_verts_features, face_areas, faces_intrinsic)
    normalised_grad_field = torch.nn.functional.normalize(diffused_verts_features_grad, p=2, dim=-1, eps=0.)
    integrated_divergence = make_integrated_divergence(d_verts_intrinsic, normalised_grad_field, cots, faces_intrinsic).to_dense()
    distance_solution = torch.linalg.solve(conformal_laplacian_kernel, integrated_divergence)
    geodesic_distance = distance_solution.subtract(distance_solution.min())
    return geodesic_distance
