<h1 align="center">SOgym</h1>
<p align="center">
  Gym environment for developing structural optimization problems using deep reinforcement learning.
</p>
<p align="center">
  <img src="https://github.com/ThomasRochefortB/sogym_v2/blob/alternative/docs/SOGYM_LOGO.png?raw=true" alt="SOgym Logo" width="200"/>
</p>
The environment is based on the topology optimization framework of Moving Morphable Components [1]. The design task is framed as a sequential decision process where at each timestep, the agent has to place one component.

## Boundary Conditions

The environment samples from the following boundary conditions distribution at the start of each episode:

<div align="center">
  <p>
    <img src="https://github.com/ThomasRochefortB/sogym_v2/blob/alternative/docs/BCs.png?raw=true" alt="Boundary Conditions Visualization" style="background-color:white; display: block; margin: auto;" width="300"/>
  </p>

  <h4>Table 1: Parameters Defining the Boundary Conditions Distribution</h3>

  <table>
    <thead>
      <tr>
        <th>Parameter</th>
        <th>Name</th>
        <th>Distribution</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>h</td>
        <td>Height</td>
        <td>[1.0, 2.0]</td>
      </tr>
      <tr>
        <td>w</td>
        <td>Width</td>
        <td>[1.0, 2.0]</td>
      </tr>
      <tr>
        <td>L_s</td>
        <td>Support Length</td>
        <td>50% to 75%</td>
      </tr>
      <tr>
        <td>P_s</td>
        <td>Support Position</td>
        <td>0 to (100% of L_s)</td>
      </tr>
      <tr>
        <td>P_L</td>
        <td>Load Position</td>
        <td>0% to 100% of boundary opposite from support</td>
      </tr>
      <tr>
        <td>θ_L</td>
        <td>Load Orientation</td>
        <td>[0°,360°] *</td>
      </tr>
    </tbody>
  </table>
</div>

*The selected angle is filtered to ensure there is at least 45 degrees of difference with the support normal.

The blue wall represents a fully supported boundary and the red boundary the region where a unit load with varying orientation is randomly placed.

The environment's reward function can be modified to fit multiple constrained topology optimization objectives such as:

* Compliance minimization under hard volume constraint [Implemented]
* Compliance minimization under soft volume constraint [Implemented]
* Compliance minimization under global/local stress constraint
* Volume minimization under compliance constraint
* Combined volume and compliance minimization

## SOgym Leaderboard

### Observation Space Configurations and Algorithms

| Observation Space | PPO           | SAC           | DreamerV3     |
|-------------------|---------------|---------------|---------------|
| Dense             | Result for PPO| Result for SAC| Result for DreamerV3|
| Image             | Result for PPO| Result for SAC| Result for DreamerV3|
| TopOpt Game       | Result for PPO| Result for SAC| Result for DreamerV3|

---
## Citation
To cite this library, please refer to the following paper:

---
## References

[1] Zhang, W., Yuan, J., Zhang, J. et al. A new topology optimization approach based on Moving Morphable Components (MMC) and the ersatz material model. Struct Multidisc Optim 53, 1243–1260 (2016). https://doi.org/10.1007/s00158-015-1372-3

[2] Nobel-Jørgensen, Morten & Malmgren-Hansen, David & Bærentzen, Andreas & Sigmund, Ole & Aage, Niels. (2016). Improving topology optimization intuition through games. Structural and Multidisciplinary Optimization. 54. 10.1007/s00158-016-1443-0. 
