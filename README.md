<h1 align="center">SOgym</h1>
<p align="center">
  Gym environment for developing structural optimization problems using deep reinforcement learning.
</p>
<p align="center">
  <img src="https://raw.githubusercontent.com/ThomasRochefortB/sogym_v2/alternative/docs/SOGYM_LOGO.png?token=GHSAT0AAAAAACRL6NQKA4ESORH7WSD22RDGZRHZR2A" alt="SOgym Logo" width="200"/>
</p>

The environment is based on the topology optimization framework of Moving Morphable Components [1]. The design task is framed as a sequential decision process where at each timestep, the agent has to place one components.

The environment consists of a 2 x 1 rectangular beam and the boundary conditions are randombly varied between episodes according to the following 6 loading cases:

![alt text](https://github.com/ThomasRochefortB/so_gym/blob/main/docs/boundary_conditions.png?raw=true)

The blue wall represents a fully supported boundary and the red boundary the region where a unit load with varying orientation is randomly placed.

The environment's reward function can be modified to fit multiple constrained topology optimization objectives such as:

* Compliance minimization under volume constraint [Implemented]
* Compliance minimization under global/local stress constraint
* Volume minimization under compliance constraint
* Combined volume and compliance minimzation



---
## Leaderboard
| **Method** | **Reward function** |                              |
|------------|---------------------|------------------------------|
|            |      **1/Comp**     | 1/Comp * ( Volume/V* -1) |
|   **PPO**  |         TBD         |              TBD             |
|   **TD3**  |         TBD         |              TBD             |


---
## References
[1] Zhang, W., Yuan, J., Zhang, J. et al. A new topology optimization approach based on Moving Morphable Components (MMC) and the ersatz material model. Struct Multidisc Optim 53, 1243–1260 (2016). https://doi.org/10.1007/s00158-015-1372-3
