import igl
import torch
from torch import Tensor


def cross_product(
    U: Tensor,
    V: Tensor
) -> Tensor:
    return (
        U[..., [1, 2, 0]].mul(V[..., [2, 0, 1]])
        .sub(U[..., [2, 0, 1]].mul(V[..., [1, 2, 0]]))
    )


def make_intrinsic_triangulation(
    edges_lengths2: Tensor,
    faces: Tensor
) -> Tensor:
    """
    Constructs an intrinsic Delaunay triangulation of a mesh given its edge lengths and
    faces.

    Args:
        edges_lengths2 (Tensor): A tensor of squared edge lengths of the mesh.
        faces (Tensor): A tensor of vertex indices for each face of the mesh.

    Returns:
        Tensor: A tensor representing the intrinsic Delaunay triangulation of the mesh,
        with duplicated vertices removed.
    """
    delaunay_array = igl.intrinsic_delaunay_triangulation(
        edges_lengths2.double().pow(1 / 2).detach().cpu().numpy(),
        faces.cpu().numpy()
    )[1]
    delaunay = torch.from_numpy(delaunay_array).to(edges_lengths2.device).long()
    # Remove faces with duplicated vertices:
    nondegen_faces = delaunay[..., [1, 2, 0]].eq(delaunay).any(dim=-1).logical_not()
    nondegen_delaunay = delaunay[nondegen_faces, :]
    return nondegen_delaunay


def make_mesh_geometry(
    edges: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Computes geometric properties of a triangle mesh, given tangent half-edge vectors.

    Args:
        edges (Tensor): A tensor of tangent half-edge vectors with shape (F, 3, 3).

    Returns:
        tuple[Tensor, Tensor, Tensor]: A tuple containing:
        - the squared length of each edge with shape (F, 3),
        - the intersection areas between faces and vertex dual cells with shape (F, 3),
        - the cotangents of the angles at each corner of each face with shape (F, 3).
    """
    edge_lengths2 = edges.pow(2).sum(dim=-1)
    dots = -edges[..., [1, 2, 0], :].multiply(edges[..., [2, 0, 1], :]).sum(dim=-1)
    crosses = cross_product(-edges[..., [1, 2, 0], :], edges[..., [2, 0, 1], :]).norm(dim=-1)
    cots = dots.div(crosses)
    voronoi_areas = edge_lengths2.multiply(cots)[..., [[1, 2], [2, 0], [0, 1]]].sum(dim=-1).div(8)
    # Use 0.25-0.25-0.5 weighting for obtuse faces:
    alternative_areas = voronoi_areas.sum(dim=-1, keepdim=True).multiply(cots.lt(0).add(1).div(4))
    cell_areas = voronoi_areas.where(cots.gt(0).all(dim=-1, keepdim=True), alternative_areas)
    return (edge_lengths2, cell_areas, cots)


def make_face_gradients(
    d_verts: Tensor,
    features: Tensor,
    face_areas: Tensor,
    faces: Tensor
) -> Tensor:
    """
    Computes gradients of the given features of a triangular mesh.

    Args:
        d_verts (Tensor): A tensor of tangent half-edge vectors with shape (F, 3, 3).
        features (Tensor): A tensor of vertex features with shape (V, C).
        face_areas (Tensor): A tensor of face areas with shape (F,).
        faces (Tensor): A tensor of vertex indices for each face of the mesh with
                        shape (F, 3).

    Returns:
        Tensor: A tensor of gradients of the features, with shape (F, C, 3).
    """
    normals = cross_product(d_verts[..., [1, 2, 0], :], d_verts[..., [2, 0, 1], :])
    tangents = cross_product(normals, d_verts)
    return (
        tangents[..., None, :]
        .multiply(features[faces, :, None])
        .sum(dim=-3)
        .div(face_areas.multiply(2).pow(2)[:, None, None])
    )


def make_differential(
    verts_features: Tensor,
    faces: Tensor
) -> Tensor:
    """
    Computes the differential operator for a triangular mesh.

    Args:
        verts_features (Tensor): A tensor of vertex features with shape (V, C).
        faces (Tensor): A tensor of vertex indices for each face with shape (F, 3).

    Returns:
        Tensor: A tensor of differential operator values with shape (F, 3, C).
    """
    faces_verts = verts_features[faces, :]
    return faces_verts[..., [2, 0, 1], :].sub(faces_verts[..., [1, 2, 0], :])


def sum_cell_values(
    num_verts: Tensor,
    cell_vals: Tensor,
    faces: Tensor
) -> Tensor:
    """
    Aggregates the values of cells to their corresponding vertices.

    Args:
        num_verts (Tensor): The number of vertices in the mesh.
        cell_vals (Tensor): The values of cells in the mesh, with shape (F, 3).
        faces (Tensor): A tensor of vertex indices for each face with shape (F, 3).

    Returns:
        Tensor: A tensor of vertex values with shape (V,).
    """
    return (
        torch.zeros((num_verts,), device=cell_vals.device, dtype=cell_vals.dtype)
        .index_add_(dim=0, index=faces.view(-1), source=cell_vals.view(-1))
    )


def make_local_cotan_weights(
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float
) -> Tensor:
    local_cotan_kernel = torch.zeros((3, 3, 3), device=device, dtype=dtype)
    local_cotan_kernel[[0, 1, 2], [0, 1, 2], [1, 2, 0]] = 1.0
    local_cotan_kernel[[0, 1, 2], [0, 1, 2], [2, 0, 1]] = 1.0
    local_cotan_kernel[[0, 1, 2], [1, 2, 0], [2, 0, 1]] = -1.0
    local_cotan_kernel[[0, 1, 2], [2, 0, 1], [1, 2, 0]] = -1.0
    return local_cotan_kernel


def make_conformal_laplacian_kernel(
    num_verts: int,
    cots: Tensor,
    faces: Tensor
) -> Tensor:
    """
    Computes and returns the conformal Laplacian kernel given the cotangent weights and
    face indices.

    Args:
    - num_verts (int): number of vertices
    - cots (Tensor): tensor of size (F, 3) of cotangents for each face
    - faces (Tensor): tensor of size (F, 3) of vertex indices for each face

    Returns:
    - Tensor: tensor of size (num_verts, num_verts) representing the conformal Laplacian
    """
    local_cotan_weights = make_local_cotan_weights(device=cots.device, dtype=cots.dtype)
    x = faces[:, [[[0,0],[0,1],[0,2]],[[1,0],[1,1],[1,2]],[[2,0],[2,1],[2,2]]]]
    return (
        torch.zeros((num_verts, num_verts), device=cots.device, dtype=cots.dtype)
        .reshape(-1)
        .index_add_(
            dim=0,
            index=(x[..., 0] * num_verts + x[..., 1]).reshape(-1),
            source=torch.einsum("ijc, fc -> fij", local_cotan_weights, cots).flatten(),
        )
        .reshape(num_verts, num_verts)
        .div(2)
    )


def diffuse_features_backward_euler(
    vertex_areas: Tensor,
    conformal_laplacian_kernel: Tensor,
    verts_features: Tensor,
    tau: Tensor
) -> Tensor:
    """
    Compute the diffusion of vertex features using the backward Euler method.

    Args:
        vertex_areas (Tensor): Vertex areas with shape (V,).
        conformal_laplacian_kernel (Tensor): Conformal Laplacian kernel with shape (V, V).
        verts_features (Tensor): Vertex features with shape (V, C).
        tau (Tensor): Time step.

    Returns:
        The updated vertex features after diffusion with shape (V, C).
    """
    return torch.linalg.solve(
        conformal_laplacian_kernel.multiply(tau).add(vertex_areas.diag()),
        verts_features.multiply(vertex_areas[..., None])
    )


def make_integrated_divergence(
    num_verts: Tensor,
    d_verts: Tensor,
    face_gradients: Tensor,
    cots: Tensor,
    faces: Tensor
) -> Tensor:
    """
    Calculates the integrated divergence of the face gradients across vertices using
    cotangents.

    Args:
    - num_verts: A tensor representing the number of vertices in the mesh.
    - d_verts: A tensor of shape (F, 3, 3) representing the differences between vertices
      of each face.
    - face_gradients: A tensor of shape (F, C, 3) representing the gradients of the
      features for each face.
    - cots: A tensor of shape (F, 3) representing the cotangents of each angle of each
      face.
    - faces: A tensor of shape (F, 3) representing the vertex indices of each face.

    Returns:
    - A tensor of shape (V, C) representing the integrated divergence of the face
      gradients across vertices.
    """
    num_features = face_gradients.shape[-2]
    idxs = faces.view(-1)
    vals = (
        d_verts[..., None, :]
        .multiply(face_gradients[..., None, :, :])
        .sum(dim=-1)
        .multiply(cots[..., None])[..., [[1, 2], [2, 0], [0, 1]], :]
        .diff(n=1, dim=-2)[..., 0, :]
        .div(2)
        .view(-1, num_features)
    )
    return (
        torch.zeros((num_verts, num_features), device=cots.device, dtype=cots.dtype)
        .index_add_(dim=0, index=idxs, source=vals)
    )


def make_geodesic_distances(
    verts: Tensor,
    faces_extrinsic: Tensor,
    verts_features: Tensor
) -> tuple[Tensor, Tensor]:
    """
    Computes geodesic distances on a triangular mesh using the heat method.

    Args:
    - verts (Tensor): A tensor of shape (num_verts, 3) representing the 3D coordinates
      of ach vertex in the mesh.
    - faces_extrinsic (Tensor): A tensor of shape (num_faces, 3) representing the
      indices of the vertices in each face of the mesh.
    - verts_features (Tensor): A tensor of shape (num_verts, num_features) containing
      the feature values at each vertex of the mesh.

    Returns:
    A tuple containing two tensors:
    - vertex_areas (Tensor): A tensor of shape (num_verts,) containing the area of each
      vertex in the mesh.
    - geodesic_distance (Tensor): A tensor of shape (num_verts,) containing the geodesic
      distance of each vertex in the mesh.

    The function first calculates the intrinsic triangulation of the mesh and the
    geometric weights and operators needed to compute geodesic distances using the heat
    method. It then applies the heat method by diffusing the input features using the
    backward Euler method, calculating the normalised gradient field and integrating the
    divergence of the normalised gradient field. Finally, it solves the Poisson equation
    to obtain the geodesic distances and returns the area of each vertex along with the
    resulting distances.
    """
    # Process geometry:
    num_verts = verts.size(0)
    ## Calculate intrinsic triangulation:
    edge_lengths2_extrinsic = make_differential(verts, faces_extrinsic).pow(2).sum(-1)
    faces_intrinsic = make_intrinsic_triangulation(
        edge_lengths2_extrinsic, faces_extrinsic
    )
    ## Calculate geometric weights and operators:
    d_verts_intrinsic = make_differential(verts, faces_intrinsic)
    (edge_lengths2_intrinsic, cell_areas, cots) = make_mesh_geometry(d_verts_intrinsic)
    face_areas = cell_areas.sum(dim=-1)
    vertex_areas = sum_cell_values(num_verts, cell_areas, faces_intrinsic)
    conformal_laplacian_kernel = make_conformal_laplacian_kernel(
        num_verts, cots, faces_intrinsic
    )
    ## Calculate diffusion parameter:
    tau = edge_lengths2_intrinsic.mean()

    # Apply heat meathod:
    ## Diffuse features:
    diffused_verts_features = diffuse_features_backward_euler(
        vertex_areas, conformal_laplacian_kernel, verts_features, tau
    )
    ## Calculate normalised gradient field:
    diffused_verts_features_grad = make_face_gradients(
        d_verts_intrinsic, diffused_verts_features, face_areas, faces_intrinsic
    )
    normalised_grad_field = torch.nn.functional.normalize(
        diffused_verts_features_grad, p=2, dim=-1
    )
    ## Integrate divergence of normalised gradient field:
    integrated_divergence = make_integrated_divergence(
        num_verts, d_verts_intrinsic, normalised_grad_field, cots, faces_intrinsic
    ).to_dense()
    distance_solution = torch.linalg.solve(
        conformal_laplacian_kernel, integrated_divergence
    )
    geodesic_distance = distance_solution.subtract(
        distance_solution.min(dim=-2, keepdim=True).values
    )
    return (vertex_areas, geodesic_distance)
