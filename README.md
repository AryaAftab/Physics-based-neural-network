# Physics-based neural network
## Motivation: 
The following sample problem will demonstrate that deep neural network can be utilized to approximate the solutions of partial differential equations (PDEs). This is a recent area of research, known as Physics Informed Neural Networks (PINNS). With this approach, a loss function is setup to penalize the fitted function’s deviation from the desired differential operator and boundary conditions. ﻿The main insight of this approach lies in the fact that the training data consists of randomly sampled points in the function’s domain. By sampling mini-batches from different parts of the domain and processing these small batches sequentially, the neural network “learns” the function without the computational bottleneck present with grid-based methods. 

<img src="https://github.com/AryaAftab/Physics-based-neural-network/blob/master/pics/Physics-based%20neural%20network.jpg" width="400" img align="right">

Problem:
Let us consider a non-linear Poisson equation:
For a domain $\Omega \subset \mathbb{R}^N$ with boundary $\partial \Omega = \Gamma_{D} \cup \Gamma_{N}$, we consider the following nonlinear Poisson equation with particular boundary
conditions reads:
![image](https://user-images.githubusercontent.com/30603302/112075012-3590c480-8b95-11eb-84e2-54732fcc61f3.png)

Here $f$ is input data and  denotes the outward directed boundary normal. The nonlinear variational form can be written in the following canonical form: find $u \in V$  such that
$$
F(u;v)=0\quad \forall\,v\in\hat{V}
$$
Here $F:V\times\hat{V}\rightarrow\mathbb{R}$  is a semilinear form, linear in the argument subsequent to the semicolon, and $V$  is some suitable function space. The semilinear form is defined as follows:
$$
F(u;v) = \int_\Omega (1 + u^2)\cdot\nabla u \cdot \nabla v - f v \,{\rm dx} = 0.
$$
To solve the nonlinear system $b(U) = 0$  by Newton’s method we compute the Jacobian A = b', where $U$  is the coefficients of the linear combination in the finite element solution $u_h = \sum_{j=1}^{N}U_j\phi_j, \;
b:\mathbb{R}^N\rightarrow\mathbb{R}^N$  and 
$$
b_i(U) = F(u_h;\hat{\phi}_i),\quad i = 1,2,\dotsc,N.
$$
Linearizing the semilinear form $F$ around $u = u_h$, we obtain 
$$
F'(u_h;\delta u,v) = \int_\Omega [(2 \delta u\nabla u_h)\cdot\nabla v + ((1+u_h^2)\nabla\delta u)\nabla v] \,{\rm dx}
$$
We note that for each fixed $u_h$,$a = F'(u_h;\,\cdot\,,\,\cdot\,)$  is a bilinear form and $L = F(u_h;\,\cdot\,,\,\cdot\,)$  is a linear form. In each Newton iteration, we thus solve a linear variational problem of the canonical form:
find $\delta u \in V_{h,0}$ such that
$$
F'(u_h;\delta u,v) = -F(u_h;v)\quad\forall\,v\in\hat{V}_h.
$$
In this demo, we shall consider the following definitions of the input function, the domain, and the boundaries:

* $\Omega = [0,1] \times [0,1]\,\,\,$  (a unit square)
* $\Gamma_{D} = \{(1, y) \subset \partial \Omega\}\,\,\,$   (Dirichlet boundary)
* $\Gamma_{N} = \{(x, 0) \cup (x, 1) \cup (0, y) \subset \partial \Omega\}\,\,\,$  (Neumann boundary)
* $f(x, y) = x\sin(y)\,\,\,$  (source term)
