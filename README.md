# Physics-based neural network with Sine Activation
## Motivation: 
The following sample problem will demonstrate that deep neural network can be utilized to approximate the solutions of partial differential equations (PDEs). This is a recent area of research, known as Physics Informed Neural Networks (PINNS). With this approach, a loss function is setup to penalize the fitted function’s deviation from the desired differential operator and boundary conditions. ﻿The main insight of this approach lies in the fact that the training data consists of randomly sampled points in the function’s domain. By sampling mini-batches from different parts of the domain and processing these small batches sequentially, the neural network “learns” the function without the computational bottleneck present with grid-based methods. 

<img src="https://github.com/AryaAftab/Physics-based-neural-network/blob/master/pics/Physics-based%20neural%20network.jpg" width="400" img align="right">

Problem:
Let us consider a non-linear Poisson equation:
<img src="http://www.sciweavers.org/tex2img.php?eq=%5C%21%5C%21%5C%21%5C%21%5C%21%5C%21%5Cnabla%5Ccdot%28%281%20%2B%20u%5E2%29%20%5Cnabla%20u%29%20%3D%20f%20%5Cquad%20%7B%5Crm%20in%7D%5C%2C%20%5COmega%2C%5C%5C%0Au%20%3D%201%20%20%5Cquad%20%20%7B%5Crm%20on%7D%5C%2C%20%5CGamma_D%2C%5C%5C%0A%5Cnabla%20u%5Ccdot%20n%20%3D%200%20%5Cquad%20%20%7B%5Crm%20on%7D%5C%2C%20%5CGamma_N&bc=Cyan&fc=Black&im=png&fs=18&ff=mathpazo&edit=0" align="center" border="0" alt="\!\!\!\!\!\!\nabla\cdot((1 + u^2) \nabla u) = f \quad {\rm in}\, \Omega,\\u = 1  \quad  {\rm on}\, \Gamma_D,\\\nabla u\cdot n = 0 \quad  {\rm on}\, \Gamma_N" width="321" height="104" />


## Prerequisites
- Tensorflow 2.x
- Matplotlib 3.3
- Numpy 1.19
- FEniCS 2019.1.0 (``` installed with conda ```)


We tested the network with two activation functions, relu and sine. The best network mode, which is close to normal, was obtained with the sine activation function.
The results are saved into the (``` pics/10*10 ```) folder.

## References
[1] Sitzmann, V., Martel, J. N., Bergman, A. W., Lindell, D. B., & Wetzstein, G. (2020). Implicit neural representations with periodic activation functions. arXiv preprint arXiv:2006.09661.<br />
[2] https://fenicsproject.org/olddocs/dolfin/1.4.0/python/demo/documented/nonlinear-poisson/python/documentation.html
