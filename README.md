<h1 align="center">SOgym</h1>
<p align="center">
  Gym environment for developing structural optimization problems using deep reinforcement learning.
</p>
<p align="center">
  <img src="https://raw.githubusercontent.com/ThomasRochefortB/sogym_v2/alternative/docs/SOGYM_LOGO.png?token=GHSAT0AAAAAACRL6NQKA4ESORH7WSD22RDGZRHZR2A" alt="SOgym Logo" width="200"/>
</p>

The environment is based on the topology optimization framework of Moving Morphable Components [1]. The design task is framed as a sequential decision process where at each timestep, the agent has to place one components.

The environment samples from the following boundary conditions distribution at the start of each episode:

<p align="center">
  <img src="https://raw.githubusercontent.com/ThomasRochefortB/sogym_v2/alternative/docs/BCs.png?token=GHSAT0AAAAAACRL6NQLAMKHI6OULSADK4QAZRH2MSA" alt="Boundary Conditions Visualization" style="background-color:white; display: block; margin: auto;" width="600"/>
</p>
### Table 1: Parameters Defining the Boundary Conditions Distribution

| Parameter | Name            | Distribution                      |
|-----------|-----------------|-----------------------------------|
| h         | Height          | [1.0, 2.0]                        |
| w         | Width           | [1.0, 2.0]                        |
| L_s       | Support Length  | 50% to 75%                        |
| P_s       | Support Position| 0 to (100% of L_s)                |
| P_L       | Load Position   | 0% to 100% of boundary opposite from support |
| θ_L       | Load Orientation| [0°,360°] *                       |

*The selected angle is filtered to ensure there is at least 45 degrees of difference with the support normal.


The blue wall represents a fully supported boundary and the red boundary the region where a unit load with varying orientation is randomly placed.

The environment's reward function can be modified to fit multiple constrained topology optimization objectives such as:

* Compliance minimization under hard volume constraint [Implemented]
* Compliance minimization under soft volume constraint [Implemented]
* Compliance minimization under global/local stress constraint
* Volume minimization under compliance constraint
* Combined volume and compliance minimzation



---
# SOgym Leaderboard

## Observation Space Configurations and Algorithms

| Observation Space | PPO           | SAC           | DreamerV3     |
|-------------------|---------------|---------------|---------------|
| Dense             | Result for PPO| Result for SAC| Result for DreamerV3|
| Image             | Result for PPO| Result for SAC| Result for DreamerV3|
| TopOpt Game       | Result for PPO| Result for SAC| Result for DreamerV3|



---
## References
[1] Zhang, W., Yuan, J., Zhang, J. et al. A new topology optimization approach based on Moving Morphable Components (MMC) and the ersatz material model. Struct Multidisc Optim 53, 1243–1260 (2016). https://doi.org/10.1007/s00158-015-1372-3
