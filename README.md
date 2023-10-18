# Differentiable Heat Method
<p align="center">
  <img src="gallery/example0.png?raw=true" width="250">
  <img src="gallery/example1.png?raw=true" width="250">
  <img src="gallery/example2.png?raw=true" width="250">
</p>

## The Heat Method
The *heat method* is an algorithm for calculating the distance between two points in a curved space. This repository implements the heat method for triangulated meshes in three-dimensional space.

### Algorithm
The input data consists of:
1. A triangulated mesh given by vertex locations $V\subset\mathbb{R}^{3}$ and a triangulation $F\subset V^{3}$.
2. A chosen vertex $v\in V$.

The output is a function $f:V\to\mathbb{R}$ that maps $u\in V$ to the geodesic distance between $u$ and $v$.

To find the geodesic distance field $f$ for the vertex $v$, we perform the following steps:
1. Diffuse the indicator function $\delta_0$ of $v$ for a small time step $t << 1$ using the backward Euler method.
2. Calculate the spatial gradient $\nabla \delta_t$ of the diffused indicator function $\delta_t$, and normalise it to length $1$, so: $\phi = \nabla \delta_t / \lvert \nabla \delta_t \rvert$.
3. We then find a solution to $f(v) = 0$ and $\nabla f = \phi$ by solving $\Delta f = \text{div }\phi $. 

This works because the solution to $f$ in $\Delta f = \text{div } \phi$ minimises the functional $\mathcal{L}(f) = \int \lvert \nabla f - \phi \rvert^2$.


### Intuitive Motivation
The intuition here is that if we place a temperature impulse at $v$, then heat will flow away from $v$ along the shortest path. This tells us the _direction_ of geoedics away from $v$. The signed distance field increases at a constant rate of $1$ unit per unit as you move away from the chosen vertex, so its gradient must have length one. 

Together, these imply that the distance field $f$ satisfies
1. $f(v)=0$,
2. $\lvert \nabla f \rvert = 1$, and
3. $\nabla f$ and $\nabla \delta_t$ are parallel for small $t$,

which turns out to be enough to determine $f$ uniquely. See reference [1] below for more details.

## Differentiability
When I implemented this I aimed to make it so PyTorch's autograd functions could backpropagate through _every_ step of the geodesic distance calculation. 

In particular, other implementations I could find calculated certain geometric operators in a way that broke the computation graph. Doing so allows them to use memory-efficient sparse linear solvers that torch doesn't have support for. By sticking with dense matrices we limit the size of the meshes, but gain the ability to optimise vertex locations based on geodesic distances. The ideal tool here would be a differentiable sparse linear solver. Adding this functionality to PyTorch is an open issue [2].

## Implementation Details

### Finite Element Discretisation
The Laplacian operator plays an important role in the above calculation, and there are a variety of ways it can be discretised. On $0$-forms, the Laplacian is given by $\Delta_0=\star_0^{-1}d^T_{1}\star_1d_0$. The differential operators are determined combinatorially based on the mesh, but we have some freedom in the $\star_0$ and $\star_1$ operators.

The first of these, $\star_0$ is a mass matrix that maps vertices to the size of their dual cell. 

$(\star_0\omega)(v^*) = \frac{\lvert v^*\rvert}{\lvert v\rvert}\omega(v) = A_v\omega(v),$

<!-- where $\lvert v \rvert = 1$ by convention, and $\lvert v^*\rvert$ is the area of its dual cell. We use mixed Voronoi-barycentric areas as described by [3] in the references below. -->

The $\star_1$ operator maps $1$-forms to $1$-forms, sending forms that measure circulation (defined on primal edges) to forms that measure flux (defined on dual edges). 
<!-- We can define $\star_1$ by its operation on $1$-forms.  -->

If $\omega$ is a $1$-form and $\sigma$ is an edge with corresponding dual edge $\sigma^*$ then

$(\star_1 \omega)(\sigma^*) = \frac{\lvert \sigma^* \rvert}{\lvert \sigma\rvert}\omega(\sigma)$.

If we use the Voronoi dual cells, the end points of the dual edge $\sigma^\star$ are the circumcentres of the triangles either side of the primal edge, and we get the formula

$\frac{\lvert \sigma^*\rvert}{\lvert \sigma\rvert} = \frac{1}{2}(\cot\alpha_{ij} + \cot \beta_{ij}),$

where $\alpha_{ij}$ and $\beta_{ij}$ are the angles of the triangle corners opposite the edge $\sigma$.


## Applications
### Geometry Optimisation
We can optimise vertex locations over the entire mesh in order to maximise, minimise, or fix the distance between arbitrary pairs of vertices.

### Surface Parameterisation
The distance function of a vertex can act as an analogue to radius in a polar coordinate system centred on it.

# TODO

I've noticed that when I use this algorithm on a triangulation of the plane, it fails to compute the right distance for vertices beyond a certain distance of the source vertex.

# Gallery
## Example Distance Fields
The images above show the geodesic distance fields from three different points on the same mesh. The chosen vertices are purple, and the mesh gets more yellow with distance.
<p align="center">
  <img src="gallery/example0.png?raw=true" width="250">
  <img src="gallery/example1.png?raw=true" width="250">
  <img src="gallery/example2.png?raw=true" width="250">
</p>

# References
1. The Heat Method for
Distance Computation, https://www.cs.cmu.edu/~kmcrane/Projects/HeatMethod/paper.pdf.
2. PyTorch Issue #69538, https://github.com/pytorch/pytorch/issues/69538.
3. Discrete Differential-Geometry Operators
for Triangulated 2-Manifolds, http://www.geometry.caltech.edu/pubs/DMSB_III.pdf.
4. Geodesics in Heat, https://arxiv.org/pdf/1204.6216.pdf

## Meshes:
3. Keenan Crane, https://www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/.
4. Thingi10K, https://ten-thousand-models.appspot.com/.