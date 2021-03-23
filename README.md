# Physics-based neural network
## Motivation: 
The following sample problem will demonstrate that deep neural network can be utilized to approximate the solutions of partial differential equations (PDEs). This is a recent area of research, known as Physics Informed Neural Networks (PINNS). With this approach, a loss function is setup to penalize the fitted function’s deviation from the desired differential operator and boundary conditions. ﻿The main insight of this approach lies in the fact that the training data consists of randomly sampled points in the function’s domain. By sampling mini-batches from different parts of the domain and processing these small batches sequentially, the neural network “learns” the function without the computational bottleneck present with grid-based methods. 

<img src="https://github.com/AryaAftab/Physics-based-neural-network/blob/master/pics/Physics-based%20neural%20network.jpg" width="400" img align="right">

Problem:
Let us consider a non-linear Poisson equation:

![equation](https://latex.codecogs.com/svg.latex?\!\!\!\!\!\!\!\!\!\nabla\cdot((1%20+%20u^2)%20\nabla%20u)%20=%20f%20\!\!\!\quad%20{\rm%20in}\,%20\Omega,\\u%20=%201%20%20\quad%20%20{\rm%20on}\,%20\Gamma_D,\\\nabla%20u\cdot%20n%20=%200%20\!\!\!\quad%20%20{\rm%20on}\,%20\Gamma_N.)
