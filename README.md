# Differentiable Geodesics
Autograd-compatible implementation of the heat method algorithm for geodesic distance calculation on a piecewise linear mesh.

This allows one to calculate geodesic distances between vertices on the mesh, and then find the gradient of this distance with respect to the vertex locations.

These gradients can be used to optimise the mesh geometry to achieve a desired distance between specified vertices.


## Example distance fields:
![alt text](https://github.com/kyle-rosa/differentiable_geodesics/blob/main/gallery/example0.png?raw=true)
![alt text](https://github.com/kyle-rosa/differentiable_geodesics/blob/main/gallery/example1.png?raw=true)
![alt text](https://github.com/kyle-rosa/differentiable_geodesics/blob/main/gallery/example2.png?raw=true)


# Based on:
- https://www.cs.cmu.edu/~kmcrane/Projects/HeatMethod/paper.pdf.
