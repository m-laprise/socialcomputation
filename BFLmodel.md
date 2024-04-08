# BFL model

## Model description

Hyperparameters:

* Number of agents $N_a$
* Number of options $N_o$
* Communication network graph:
  * $\mathcal{G}_a$ with vertices $\mathcal{V}_a$ and weighted adjacency matrix $A_a=[a^a_{ik}]\in\mathbb{R}^{N_a \times N_a}$
* Belief system graph:
  * $\mathcal{G}_o$ with vertices $\mathcal{V}_o$ and weighted adjacency matrix $A_o=[a^o_{jl}]\in\mathbb{R}^{N_o \times N_o}$
* Attention network graph:
  * $\mathcal{G}_u$ with vertices $\mathcal{V}_a$ and weighted adjacency matrix $A_u=[a^u_{ik}]\in\mathbb{R}^{N_a \times N_a}$
  * Possible simplification is making it the same as the communication graph.

Model of continuous time updates of $z_{ij}$ and $u_i$ as a function of time $t \in \mathbb{R}_{\geq0}$ (coupled ODEs for rate of change).

### Opinion

Opinion of agent $i$ on topic $j$ for all $i \in \mathcal{V}_a$, $j \in \mathcal{V}_o$:

$$\dot{z}_{ij} = -d_{ij}z_{ij} + S \left( u_i \left( \alpha^j_i z_{ij} + \sum^{N_a}_{k\neq i} a^a_{ik}z_{kj} + \sum^{N_o}_{l\neq j} a^o_{jl}z_{il} + \sum^{N_a}_{k\neq i} \sum^{N_o}_{l\neq j} a^a_{ik}a^o_{jl}z_{kl}\right) \right) + b_{ij}$$

Terms are defined as follows:

* The first term is linear negative feedback, the second term is nonlinear positive feedback, and the last term is input (stimuli or bias)
* $d_{ij} > 0$ is a damping coefficient 
* $S: \mathbb{R} \rightarrow \mathbb{R}$ is a bounded saturation function such that:
  * $S(0)=0$
  * $S'(0)=1$
  * $S'''(0)\neq0$
* $u_i$ is the attention of agent $i$, it introduces the attention modulation

### Attention

Attention of agent $i$:
$$ \tau_u \dot{u}_i = -u_i+u_0+K_u \sum ^{N_a}_{k=1} \sum^{N_o}_{j=1} a^u_{ik}(z_{kj})^2$$

Terms are defined as follows:

* $\tau_u \geq 0$ is a time-constant
* $u_0$ is a basal level of attention
* $K_u \geq 0$ is an attention gain

### System state

State vectors:

* System opinion state: $Z =(z_i, \dots, z_{N_a}) \in \mathbb{R}^{N_a N_o}$
* ...