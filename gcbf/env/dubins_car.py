import torch
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import pybullet as p
import pybullet_data

from torch import Tensor
from typing import Tuple, Optional, Union, List
from cvxpy import Expression
from torch_geometric.data import Data, Batch
from torch_geometric.utils import mask_to_index, index_to_mask

from .simple_car import SimpleCar
from .utils import plot_graph


class DubinsCar(SimpleCar):
    """
    State: [x, y, theta, v_x, v_y]

    demo 0: agents cross the intersection with obstacles in pybullet
    demo 1: agents in no obstacle environments in pybullet
    demo 2: limit the maximum travelling distance
    demo 3: agents cross the intersection with static large obstacles and small moving obstacles in pybullet
    """

    def __init__(
            self,
            num_agents: int,
            device: torch.device,
            dt: float = 0.03,
            params: dict = None,
            max_neighbors: Optional[int] = None
    ):
        super(DubinsCar, self).__init__(num_agents, device, dt, params, max_neighbors)
        self._ref_path = None
        self._num_obs = self._params['num_obs']
        self._params['obs_len_max'] = self._params['area_size'] / 8.0
        self._obs = None  # obstacle point cloud
        self._obs_vertices = None  # obstacle vertices

        # pybullet
        self._physics_client = None
        self._plane_id = None
        self._obs_id = []
        self._agent_id = []
        self._goal_id = []
        self._projection_matrix = None
        self._view_matrix = None
        self._obs_v = None  # [theta, v]
        self._lidar_id = []
        self._link_id = []

    def demo(self, idx: int):
        self._mode = f'demo_{idx}'

        if idx == 0 or idx == 1 or idx == 3:
            # use pybullet environment and LiDAR for testing
            self._physics_client = p.connect(p.DIRECT)
            # self._physics_client = p.connect(p.GUI)
            p.setTimeStep(self._dt)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, 0.)
            visual_shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[1000, 1000, 0.001], rgbaColor=[1, 1, 1, 1])
            self._plane_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=visual_shape_id,
                basePosition=[0, 0, -0.001],
                baseOrientation=[0, 0, 0, 1]
            )
            if idx == 3:
                self._params['obs_len_max'] = self._params['car_radius'] * 2

    @property
    def max_episode_steps(self) -> int:
        if self._mode == 'train':
            return 500
        elif self._mode == 'test' or self._mode == 'demo_2':
            return 2500
        elif self._mode == 'demo_0' or self._mode == 'demo_3':
            return 2000
        elif self._mode == 'demo_1':
            return 2500

    @property
    def default_params(self) -> dict:
        return {
            'max_distance': 4.0,  # maximum moving distance to goal
            'area_size': 4.0,
            'car_radius': 0.05,
            'dist2goal': 0.05,
            'comm_radius': 1.0,
            'obs_point_r': 0.05,
            'obs_len_max': 0.5,
            'speed_limit': 0.8,
            'obs_speed_limit': 0.2,
            'num_obs': 0,
        }

    @property
    def state_dim(self) -> int:
        return 4

    @property
    def edge_dim(self) -> int:
        return 5

    def dynamics(self, data: Data, u: Union[Tensor, Expression]) -> Union[Tensor, Expression]:
        if isinstance(u, Expression):
            raise NotImplementedError
        else:
            agent_mask = data.agent_mask
            xdot = torch.zeros_like(data.states)
            xdot[:, 0] = torch.clamp(data.states[:, 3], max=self._params['speed_limit']) * torch.cos(data.states[:, 2])
            xdot[:, 1] = torch.clamp(data.states[:, 3], max=self._params['speed_limit']) * torch.sin(data.states[:, 2])
            xdot[agent_mask, 2] = u[:, 0] * 10
            xdot[agent_mask, 3] = u[:, 1]

            # speed limit
            over_speed = torch.where(torch.norm(xdot[agent_mask, :2], dim=1) > self._params['speed_limit'])[0]
            if over_speed.shape[0] > 0:
                xdot[agent_mask, 3][over_speed] = 0

            if data.states.shape[0] == self.num_agents + self._obs.shape[0]:
                reach = torch.less(torch.norm(data.states[agent_mask, :2] - self._goal[:, :2], dim=1),
                                   self._params['dist2goal'])
                xdot[agent_mask] *= torch.logical_not(reach).unsqueeze(1).repeat(1, self.state_dim)
                return xdot
            else:
                return xdot

    def _reset_bullet(self):
        for i in self._obs_id:
            p.removeBody(i)
        self._obs_id.clear()

        for i in self._agent_id:
            p.removeBody(i)
        self._agent_id.clear()

        for i in self._goal_id:
            p.removeBody(i)
        self._goal_id.clear()

    def _init_obs_bullet(self, n: int):
        mass = 0  # make it static
        self._obs_v = torch.rand(n, 2)
        if self._mode == 'demo_3':
            self._obs_v = torch.zeros(n + 4, 2)
        for i in range(n):
            center = np.random.rand(3) * self._params['area_size'] / 4
            if i % 2 == 0:
                center[0] += self._params['area_size'] / 2 - self._params['area_size'] / 8
                center[1] *= 4
                theta = np.pi / 2
            else:
                center[1] += self._params['area_size'] / 2 - self._params['area_size'] / 8
                center[0] *= 4
                theta = 0
            center[2] = 0  # height
            length = np.random.rand() * self._params['obs_len_max'] + self._params['area_size'] / 80
            width = self._params['area_size'] / 80
            size = np.array([length, width, 0.1])

            visual_shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=size / 2, rgbaColor=[139 / 256, 0, 0, 1])
            collision_shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=size / 2)
            orientation = p.getQuaternionFromEuler([0, 0, theta])
            box_id = p.createMultiBody(baseMass=100 if i % 2 == 0 else 0.1,
                                       baseCollisionShapeIndex=collision_shape_id,
                                       baseVisualShapeIndex=visual_shape_id,
                                       basePosition=center,
                                       baseOrientation=orientation)
            valid_obs = True
            if self._mode == 'demo_3':
                # check collision
                for obs in self._obs_id:
                    closest_points = p.getClosestPoints(
                        box_id, obs, self._params['obs_point_r'])
                    if len(closest_points) > 0:
                        p.removeBody(box_id)
                        valid_obs = False
                        break
            if valid_obs:
                self._obs_id.append(box_id)
                self._obs_v[i, 1] = (2 * torch.rand(1) - 1) * self._params['obs_speed_limit']
                self._obs_v[i, 0] = theta
        if self._mode == 'demo_3':
            # generate some large static obstacles
            square_size = self._params['area_size'] / 16 * 3
            centers = np.array([[square_size, square_size, 0],
                                [square_size, self._params['area_size'] - square_size, 0],
                                [self._params['area_size'] - square_size, square_size, 0],
                                [self._params['area_size'] - square_size, self._params['area_size'] - square_size, 0]])
            for j, center in enumerate(centers):
                size = np.array([square_size / 2, square_size / 4, 0.5])
                visual_shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=size / 2, rgbaColor=[139 / 256, 0, 0, 1])
                collision_shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=size / 2)
                orientation = p.getQuaternionFromEuler([0, 0, 0])
                box_id = p.createMultiBody(baseMass=0,
                                           baseCollisionShapeIndex=collision_shape_id,
                                           baseVisualShapeIndex=visual_shape_id,
                                           basePosition=center,
                                           baseOrientation=orientation)
                self._obs_id.append(box_id)
                self._obs_v[n + j, 1] = 0
                self._obs_v[n + j, 0] = 0

    def _init_agent_bullet(self, n: int, obs_id: List[int], pos_mode: str):
        mass = 0
        side_length = self._params['area_size']

        # initialize positions
        pos = np.zeros((self.num_agents, 3))
        i = 0
        while i < n:
            if pos_mode == 'random':
                candidate = np.zeros(3)
                candidate[:2] = np.random.rand(2) * side_length
            elif pos_mode == 'cross':
                candidate = np.zeros(3)
                candidate[:2] = np.random.rand(2) * side_length / 8 * 3
                if i % 2 == 0:
                    candidate[0] += side_length / 16 * 13 - side_length / 16 * 3
                if self._mode == 'demo_3':
                    if i % 4 == 3 or i % 4 == 0:
                        candidate[1] += side_length / 16 * 13 - side_length / 16 * 3
            else:
                raise NotImplementedError

            dist_min = np.min(np.linalg.norm(pos - candidate, axis=1))
            if dist_min <= self._params['car_radius'] * 4:
                continue
            pos[i] = candidate

            # create agents
            car_id = p.loadURDF(
                'racecar/racecar.urdf',
                pos[i],
                p.getQuaternionFromEuler([0, 0, np.random.rand() * 2 * torch.pi]),
                globalScaling=0.5  # 0.2
            )

            # agents cannot be too close to obstacles
            valid_agent = True
            for obs in obs_id:
                closest_points = p.getClosestPoints(
                    car_id, obs, self._params['car_radius'] * 2 + self._params['obs_point_r'] * 2)
                if len(closest_points) > 0:
                    p.removeBody(car_id)
                    valid_agent = False
                    break
            if valid_agent:
                i += 1
                self._agent_id.append(car_id)

    def _init_goal_bullet(self, n: int, obs_id: List[int], pos_mode: str):
        mass = 0  # make it static
        side_length = self._params['area_size']

        # initialize positions
        pos = np.zeros((self.num_agents, 3))
        i = 0
        while i < n:
            if pos_mode == 'random':
                candidate = np.zeros(3)
                candidate[:2] = np.random.rand(2) * side_length
            elif pos_mode == 'cross':
                candidate = np.zeros(3)
                candidate[:2] = np.random.rand(2) * side_length / 8 * 3
                if i % 2 == 1:
                    candidate[0] += side_length / 16 * 13
                candidate[1] += side_length / 16 * 13
                if self._mode == 'demo_3':
                    if i % 4 == 3 or i % 4 == 0:
                        candidate[1] -= side_length / 16 * 13
            else:
                raise NotImplementedError

            dist_min = np.min(np.linalg.norm(pos - candidate, axis=1))
            if dist_min <= self._params['car_radius'] * 8:
                continue
            pos[i] = candidate

            # create goals with cylinders
            collision_shape_id = p.createCollisionShape(
                p.GEOM_CYLINDER, radius=self._params['car_radius'], height=0.05
            )
            visual_shape_id = p.createVisualShape(
                p.GEOM_CYLINDER, radius=self._params['car_radius'], length=0.05,
                rgbaColor=[30 / 256, 132 / 256, 73 / 256, 1])
            orientation = p.getQuaternionFromEuler([0, 0, np.random.rand() * 2 * torch.pi])
            cylinder_id = p.createMultiBody(baseMass=mass,
                                            baseCollisionShapeIndex=collision_shape_id,
                                            baseVisualShapeIndex=visual_shape_id,
                                            basePosition=pos[i],
                                            baseOrientation=orientation)

            # agents cannot be too close to obstacles
            valid_goal = True
            for obs in obs_id:
                closest_points = p.getClosestPoints(
                    cylinder_id, obs,
                    self._params['car_radius'] * 2 + self._params['obs_point_r'] * 2
                )
                if len(closest_points) > 0:
                    p.removeBody(cylinder_id)
                    valid_goal = False
                    break
            if valid_goal:
                i += 1
                self._goal_id.append(cylinder_id)

        if self._mode == 'demo_3':
            np.random.shuffle(self._goal_id)

    def _lidar(self, agent_id: int) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[None, None]]:
        n_rays = 32

        agent_pos = np.array(p.getBasePositionAndOrientation(agent_id)[0])

        # calculate ray end positions
        ray_end_pos = []
        for j in range(n_rays):
            theta = j * 2 * np.pi / n_rays
            end_pos = np.array([np.cos(theta), np.sin(theta), 0]) * self._params['comm_radius'] + agent_pos
            ray_end_pos.append(end_pos)

        lidar_results = list(p.rayTestBatch(
            rayFromPositions=np.repeat(agent_pos.reshape(1, 3), n_rays, axis=0),
            rayToPositions=ray_end_pos
        ))

        hit_position = []
        hit_velocity = []
        for i in reversed(range((len(lidar_results)))):
            if lidar_results[i][0] in self._obs_id:
                hit_position.append(np.array(lidar_results[i][3]))
                hit_velocity.append(self._obs_v[self._obs_id.index(lidar_results[i][0])])

        # draw lidar rays
        for i in range(len(hit_position)):
            visual_shape_id = p.createVisualShape(
                p.GEOM_CYLINDER, radius=0.005, length=np.linalg.norm(hit_position[i] - agent_pos),
                rgbaColor=[238 / 256, 163 / 256, 51 / 256, 1]
            )
            ray_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=visual_shape_id,
                basePosition=(hit_position[i] + agent_pos) / 2,
                baseOrientation=p.getQuaternionFromEuler([0, np.pi / 2, np.arctan2(
                    hit_position[i][1] - agent_pos[1], hit_position[i][0] - agent_pos[0])])
            )
            self._lidar_id.append(ray_id)

        if len(hit_position) == 0:
            return None, None
        else:
            return np.stack(hit_position), np.stack(hit_velocity)

    def _get_observation_bullet(self):
        obs_pos = []
        obs_vel = []
        for i in self._agent_id:
            hit_pos, hit_vel = self._lidar(i)
            if hit_pos is not None:
                obs_pos.append(hit_pos)
                obs_vel.append(hit_vel)
        if len(obs_pos) > 0:
            obs_pos = np.concatenate(obs_pos, axis=0)[:, :2]
            obs_vel = np.concatenate(obs_vel, axis=0)
        else:
            obs_pos = np.zeros((0, 2))
            obs_vel = np.zeros((0, 2))
        self._obs = torch.zeros(obs_pos.shape[0], self.state_dim, device=self.device)
        obs_pos = torch.from_numpy(obs_pos).type_as(self._obs)
        obs_vel = torch.from_numpy(obs_vel).type_as(self._obs)
        self._obs[:, :2] = obs_pos
        self._obs[:, 2:] = obs_vel
        return obs_pos

    def reset(self) -> Data:
        self._t = 0

        side_length = self._params['area_size']
        states = torch.zeros(self.num_agents, 2, device=self.device)
        goals = torch.zeros(self.num_agents, 2, device=self.device)

        if self._mode == 'train' or self._mode == 'test' or self._mode == 'demo_2':
            # generate obstacles
            i = 0
            obs_pos = torch.zeros(self._num_obs, 2, device=self.device)
            while i < self._num_obs:
                obs_pos[i] = torch.rand(2, device=self.device) * side_length
                i += 1
            self._obs = torch.rand(obs_pos.shape[0], self.state_dim, device=self.device)
            self._obs[:, :2] = obs_pos
            self._obs[:, 2] *= torch.pi * 2
            self._obs[:, 3] *= self._params['obs_speed_limit']

            # randomly generate positions of agents
            i = 0
            while i < self.num_agents:
                candidate = torch.rand(2, device=self.device) * side_length
                dist_min = torch.norm(states - candidate, dim=1).min()
                if dist_min <= self._params['car_radius'] * 4:
                    continue
                dist_min = torch.norm(obs_pos - candidate, dim=1)
                if dist_min.numel() == 0:
                    pass
                else:
                    dist_min = dist_min.min()
                    if dist_min <= self._params['car_radius'] * 2 + self._params['obs_point_r'] * 2:
                        continue
                states[i] = candidate
                i += 1

            # randomly generate goals of agents
            i = 0
            while i < self.num_agents:
                if self._mode == 'demo_2':
                    candidate = (torch.rand(2, device=self.device) * 2 - 1) * self._params['max_distance'] + states[i]
                    if (candidate > self._params['area_size']).any() or (candidate < 0).any():
                        continue
                else:
                    candidate = torch.rand(2, device=self.device) * side_length
                dist_min = torch.norm(goals - candidate, dim=1).min()
                if dist_min <= self._params['car_radius'] * 5:
                    continue
                dist_min = torch.norm(obs_pos - candidate, dim=1)
                if dist_min.numel() != 0:

                    if dist_min.min() <= self._params['car_radius'] * 2 + self._params['obs_point_r'] * 2:
                        continue
                goals[i] = candidate
                i += 1

            # add velocity and heading
            states = torch.cat([states, torch.zeros(self.num_agents, 2, device=self.device)], dim=1)
            states[:, 2] = torch.rand_like(states[:, 3]) * 2 * torch.pi - torch.pi

            # record goals
            goals = torch.cat([goals, torch.zeros(self.num_agents, 2, device=self.device)], dim=1)
            goals[:, 2] = torch.rand_like(goals[:, 3]) * 2 * torch.pi - torch.pi
            self._goal = goals

        elif self._mode == 'demo_0' or self._mode == 'demo_1' or self._mode == 'demo_3':
            self._reset_bullet()
            if self._mode == 'demo_0' or self._mode == 'demo_3':
                self._init_obs_bullet(self._num_obs)
                self._init_agent_bullet(self.num_agents, self._obs_id, pos_mode='cross')
                self._init_goal_bullet(self.num_agents, self._obs_id, pos_mode='cross')
            elif self._mode == 'demo_1':
                self._init_agent_bullet(self.num_agents, self._obs_id, pos_mode='random')
                self._init_goal_bullet(self.num_agents, self._obs_id, pos_mode='random')

            # get LiDAR output
            obs_pos = self._get_observation_bullet()

            # get initial states
            states = torch.zeros(self.num_agents, self.state_dim, device=self.device)
            for i in range(len(self._agent_id)):
                pos = p.getBasePositionAndOrientation(self._agent_id[i])[0]
                theta = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self._agent_id[i])[1])[2]
                states[i] = torch.tensor([pos[0], pos[1], theta, 0.]).type_as(states)

            # get goals
            goals = torch.zeros(self.num_agents, 4, device=self.device)
            for i in range(len(self._goal_id)):
                pos = p.getBasePositionAndOrientation(self._goal_id[i])[0]
                goals[i] = torch.tensor([pos[0], pos[1], 0., 0.]).type_as(states)
            self._goal = goals
        else:
            raise ValueError('Reset environment: unknown type of mode!')

        # build graph
        data = Data(
            x=torch.cat(
                (torch.zeros(self.num_agents, self.node_dim), torch.ones(self._obs.shape[0], self.node_dim)), dim=0
            ).type_as(states),
            pos=torch.cat((states[:, :2], obs_pos), dim=0).type_as(states),
            states=torch.cat((states, self._obs), dim=0).type_as(states),
            agent_mask=index_to_mask(torch.arange(self.num_agents, device=self.device),
                                     size=states.shape[0] + self._obs.shape[0])
        )
        data = self.add_communication_links(data)
        self._data = data

        # set parameters for plotting
        points = torch.cat([states[:, :2], goals[:, :2], obs_pos], dim=0).cpu().detach().numpy()
        xy_min = np.min(points, axis=0) - self._params['car_radius'] * 5
        xy_max = np.max(points, axis=0) + self._params['car_radius'] * 5
        max_interval = (xy_max - xy_min).max()
        self._xy_min = xy_min - 0.5 * (max_interval - (xy_max - xy_min))
        self._xy_max = xy_max + 0.5 * (max_interval - (xy_max - xy_min))

        if self._mode == 'demo_0' or self._mode == 'demo_1' or self._mode == 'demo_3':
            # set up camera
            view_angle = np.pi / 3
            camera_dist = max_interval * 1.5
            target_pos = np.array([(xy_min[0] + xy_max[0]) / 2, (xy_min[1] + xy_max[1]) / 2, 0])
            camera_pos = np.array([0, (xy_min[1] + xy_max[1]) / 2, 0])
            camera_pos[2] = np.linalg.norm(target_pos[:2] - camera_pos[:2]) * np.tan(view_angle)
            camera_dir = (camera_pos - target_pos) / np.linalg.norm(camera_pos - target_pos)
            camera_pos = target_pos + camera_dir * camera_dist
            self._projection_matrix = p.computeProjectionMatrixFOV(
                fov=50,
                aspect=1280. / 1280.,
                nearVal=0.1,
                farVal=50
            )
            self._view_matrix = p.computeViewMatrix(
                cameraEyePosition=camera_pos,
                cameraTargetPosition=target_pos,
                cameraUpVector=[0, 0, 1]
            )

        return data

    def step(self, action: Tensor) -> Tuple[Data, float, bool, dict]:
        # remove lidar rays
        if self._mode == 'demo_0' or self._mode == 'demo_1' or self._mode == 'demo_3':
            for i in range(len(self._lidar_id)):
                p.removeBody(self._lidar_id[i])
            self._lidar_id = []
            for i in range(len(self._link_id)):
                p.removeBody(self._link_id[i])
            self._link_id = []

        self._t += 1

        # calculate next state using dynamics
        reward_action = -torch.norm(action, dim=1).sum() * 0.01
        action = action + self.u_ref(self._data)
        lower_lim, upper_lim = self.action_lim
        action = torch.clamp(action, lower_lim, upper_lim)
        prev_reach = torch.less(torch.norm(self.data.states[self.data.agent_mask, :2] - self._goal[:, :2], dim=1),
                                self._params['dist2goal'])
        with torch.no_grad():
            state = self.forward(self._data, action)

        # in testing mode, set positions of agents and obstacles
        if self._mode == 'demo_0' or self._mode == 'demo_1' or self._mode == 'demo_3':
            for i in range(len(self._agent_id)):
                p.resetBasePositionAndOrientation(
                    self._agent_id[i],
                    [state[i, 0].item(), state[i, 1].item(), 0.],
                    p.getQuaternionFromEuler([0., 0., state[i, 2].item()])
                )
            if self._obs_v is not None:
                if self._mode == 'demo_3':
                    # simulate using physical engine
                    obs_v = self._obs_v.cpu().numpy()
                    for i in range(len(self._obs_id)):
                        p.resetBaseVelocity(self._obs_id[i],
                                            [obs_v[i, 1] * np.cos(obs_v[i, 0]), obs_v[i, 1] * np.sin(obs_v[i, 0]), 0.])
                else:
                    obs_v = self._obs_v.cpu().numpy()
                    for i in range(len(self._obs_id)):
                        obs_pos = np.array(p.getBasePositionAndOrientation(self._obs_id[i])[0])[:2]
                        obs_pos += np.array(
                            [obs_v[i, 1] * np.cos(obs_v[i, 0]), obs_v[i, 1] * np.sin(obs_v[i, 0])]) * self.dt
                        p.resetBasePositionAndOrientation(
                            self._obs_id[i],
                            [obs_pos[0], obs_pos[1], 0.],
                            p.getQuaternionFromEuler([0., 0., self._obs_v[i, 0]])
                        )
        if self._mode == 'demo_3':
            p.stepSimulation()

        # construct graph using the new states
        if self._mode == 'train' or self._mode == 'test' or self._mode == 'demo_1' or self._mode == 'demo_2':
            data = Data(
                x=torch.cat(
                    (torch.zeros(self.num_agents, self.node_dim), torch.ones(self._obs.shape[0], self.node_dim)), dim=0
                ).type_as(state),
                pos=state[:, :2],
                states=state,
                agent_mask=self._data.agent_mask
            )
        elif self._mode == 'demo_0' or self._mode == 'demo_3':
            # get LiDAR output
            obs_pos = self._get_observation_bullet()
            agent_state = state[self._data.agent_mask]
            data = Data(
                x=torch.cat(
                    (torch.zeros(self.num_agents, self.node_dim), torch.ones(self._obs.shape[0], self.node_dim)), dim=0
                ).type_as(state),
                pos=torch.cat((agent_state[:, :2], obs_pos), dim=0).type_as(agent_state),
                states=torch.cat((agent_state, self._obs), dim=0).type_as(agent_state),
                agent_mask=index_to_mask(torch.arange(self.num_agents, device=self.device),
                                         size=self._num_agents + self._obs.shape[0])
            )
        else:
            raise ValueError('Step environment: unknown type of mode!')
        self._data = self.add_communication_links(data)

        # the episode ends when reaching max_episode_steps or all the agents reach the goal
        time_up = self._t >= self.max_episode_steps
        reach = torch.less(torch.norm(self.data.states[self.data.agent_mask, :2] - self._goal[:, :2], dim=1),
                           self._params['dist2goal'])
        done = time_up or reach.all()

        # reward function
        reward_step = -0.0001
        reward_collision = -self.collision_mask(data).int() * 0.1
        reward_reach = (reach.int() - prev_reach.int()).int() * 10
        reward = reward_reach + reward_collision + reward_step + reward_action

        safe = float(1.0 - self.collision_mask(data).sum() / self.num_agents)
        collision_agent = torch.where(self.collision_mask(data) > 0)[0]
        return self.data, reward.cpu().detach().numpy(), done, {'reach': reach, 'collision': collision_agent,
                                                                'safe': safe}

    def forward_graph(self, data: Data, action: Tensor) -> Data:
        # calculate next state using dynamics
        action = action + self.u_ref(data)
        lower_lim, upper_lim = self.action_lim
        action = torch.clamp(action, lower_lim, upper_lim)
        state = self.forward(data, action)
        edge_attr = self.edge_attr(state, data.edge_index)

        # construct the graph of the next step, retaining the connection
        data_next = Data(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=edge_attr,
            pos=state[:, :2],
            states=state,
            agent_mask=data.agent_mask
        )

        return data_next

    def render(
            self, traj: Optional[Tuple[Data, ...]] = None, return_ax: bool = False, plot_edge: bool = True, ax=None
    ) -> Union[Tuple[np.array, ...], np.array]:
        return_tuple = True
        gif = []

        if traj is None:
            return_tuple = False
            if self._mode == 'train' or self._mode == 'test' or self._mode == 'demo_2':
                data = self.data
                traj = (data,)
            elif self._mode == 'demo_0' or self._mode == 'demo_1' or self._mode == 'demo_3':
                # # add communication edges
                for i in range(self.data.edge_index.shape[1]):
                    if self.data.edge_index[0, i] < self.num_agents:
                        pos1 = np.array(p.getBasePositionAndOrientation(self._agent_id[self.data.edge_index[0, i]])[0])
                        pos2 = np.array(p.getBasePositionAndOrientation(self._agent_id[self.data.edge_index[1, i]])[0])
                        visual_shape = p.createVisualShape(
                            shapeType=p.GEOM_CYLINDER,
                            radius=0.01,
                            length=np.linalg.norm(pos1 - pos2),
                            rgbaColor=[26 / 256, 82 / 256, 118 / 256, 1]
                        )
                        link_id = p.createMultiBody(
                            baseMass=0,
                            baseCollisionShapeIndex=-1,
                            baseVisualShapeIndex=visual_shape,
                            basePosition=(pos1 + pos2) / 2,
                            baseOrientation=p.getQuaternionFromEuler(
                                [0, np.pi / 2, np.arctan2(pos2[1] - pos1[1], pos2[0] - pos1[0])])
                        )
                        self._link_id.append(link_id)

                width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
                    width=1280,
                    height=1280,
                    viewMatrix=self._view_matrix,
                    projectionMatrix=self._projection_matrix,
                    renderer=p.ER_BULLET_HARDWARE_OPENGL,
                    shadow=0,
                    lightDirection=[0, 0, 1],
                )
                rgb_img = np.reshape(rgb_img, (height, width, 4))[:, :, :3]
                gif.append(rgb_img)
            else:
                raise NotImplementedError

        if self._mode == 'train' or self._mode == 'test' or self._mode == 'demo_2':

            r = self._params['car_radius']

            for data in traj:
                if ax is None:
                    fig, ax = plt.subplots(1, 1, figsize=(12, 10), dpi=100)

                # plot the cars and the communication network
                plot_graph(ax, data, radius=r, color='#FF8C00', with_label=False,
                           plot_edge=plot_edge, alpha=0.8)

                # plot the goals
                goal_data = Data(pos=self._goal[:, :2])
                plot_graph(ax, goal_data, radius=r, color='#3CB371',
                           with_label=True, plot_edge=False, alpha=0.8)

                # set axis limit
                x_interval = self._xy_max[0] - self._xy_min[0]
                y_interval = self._xy_max[1] - self._xy_min[1]
                ax.set_xlim(self._xy_min[0], self._xy_min[0] + max(x_interval, y_interval))
                ax.set_ylim(self._xy_min[1], self._xy_min[1] + max(x_interval, y_interval))
                plt.axis('off')
                plt.tight_layout()

                if return_ax:
                    return ax

                # convert to numpy array
                fig.canvas.draw()
                fig_np = np.frombuffer(
                    fig.canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))
                gif.append(fig_np)
                plt.close()

        if return_tuple:
            return tuple(gif)
        else:
            return gif[0]

    def edge_attr(self, state: Tensor, edge_index: Tensor) -> Tensor:
        edge_info = torch.cat([state[:, :3],
                               (state[:, 3] * torch.cos(state[:, 2])).unsqueeze(1),
                               (state[:, 3] * torch.sin(state[:, 2])).unsqueeze(1)], dim=1)
        return edge_info[edge_index[0]] - edge_info[edge_index[1]]

    def add_communication_links(self, data: Data) -> Data:
        pos_diff = data.pos.unsqueeze(1) - data.pos.unsqueeze(0)  # [i, j]: j -> i
        dist = torch.norm(pos_diff, dim=-1)[data.agent_mask]
        dist[:, :dist.shape[0]] += torch.eye(dist.shape[0], device=self.device) * (self._params['comm_radius'] + 1)

        # filter out top k neighbors
        if self._max_neighbors is not None:
            _, dist_id = torch.topk(dist, self._max_neighbors, dim=-1, largest=False)
            for i in range(dist_id.shape[0]):
                neighbor_mask = index_to_mask(dist_id[i], dist.shape[1])
                dist[i, ~neighbor_mask] += self._params['comm_radius'] + 1

        dist_mask = torch.less(dist, self._params['comm_radius'])
        edge_index = torch.nonzero(dist_mask, as_tuple=False).t()[[1, 0]]
        edge_attr = self.edge_attr(data.states, edge_index)
        data.update(Data(edge_index=edge_index, edge_attr=edge_attr))
        return data

    @property
    def state_lim(self) -> Tuple[Tensor, Tensor]:
        low_lim = torch.tensor(
            [self._xy_min[0], self._xy_min[1], -10, -10],
            device=self.device)
        high_lim = torch.tensor(
            [self._xy_max[0], self._xy_max[1], 10, 10],
            device=self.device)
        return low_lim, high_lim

    @property
    def action_lim(self) -> Tuple[Tensor, Tensor]:
        upper_limit = torch.ones(2, device=self.device) * 2.
        lower_limit = - upper_limit
        return lower_limit, upper_limit

    def u_ref(self, data: Data) -> Tensor:
        states = data.states[data.agent_mask]
        states = states.reshape(-1, self.num_agents, self.state_dim)
        diff = (states - self._goal).reshape(-1, self.state_dim)
        states = states.reshape(-1, self.state_dim)

        # PID parameters
        k_omega = 0.2  # 0.2
        k_v = 0.3
        k_a = 0.6  # 0.6

        dist = torch.norm(diff[:, :2], dim=-1)
        theta_t = (torch.acos(-diff[:, 0] / (dist + 0.0001)) * torch.sign(-diff[:, 1])) % (2 * torch.pi)
        theta = states[:, 2] % (2 * torch.pi)
        theta_diff = theta_t - theta
        omega = torch.zeros(states.shape[0]).type_as(states)
        agent_dir = torch.cat([torch.cos(theta).unsqueeze(-1), torch.sin(theta).unsqueeze(-1)], dim=-1)
        theta_between = torch.acos(
            torch.clamp(torch.bmm(-diff[:, :2].unsqueeze(1), agent_dir.unsqueeze(-1)).squeeze() / (dist + 0.0001), -1,
                        1))

        # when theta <= pi
        small_anti_clock_id = torch.where(
            torch.logical_and(torch.logical_and(theta_diff < torch.pi, theta_diff >= 0), theta <= torch.pi))
        small_clock_id = torch.where(
            torch.logical_and(torch.logical_not(torch.logical_and(theta_diff < torch.pi, theta_diff >= 0)),
                              theta <= torch.pi))
        omega[small_anti_clock_id] = k_omega * theta_between[small_anti_clock_id]
        omega[small_clock_id] = -k_omega * theta_between[small_clock_id]

        # when theta > pi
        large_clock_id = torch.where(
            torch.logical_and(torch.logical_and(theta_diff > -torch.pi, theta_diff <= 0), theta > torch.pi))
        large_anti_clock_id = torch.where(
            torch.logical_and(torch.logical_not(torch.logical_and(theta_diff > -torch.pi, theta_diff <= 0)),
                              theta > torch.pi))
        omega[large_clock_id] = -k_omega * theta_between[large_clock_id]
        omega[large_anti_clock_id] = k_omega * theta_between[large_anti_clock_id]
        omega = torch.clamp(omega, -5., 5.)

        a = -k_a * states[:, 3] + k_v * torch.norm(diff[:, :2], dim=-1)
        over_speed_agent = torch.where(states[:, 3] - self._params['speed_limit'] > 0)[0]
        if over_speed_agent.shape[0] > 0:
            a[over_speed_agent] = torch.clamp(a[over_speed_agent], max=0.)
        over_speed_agent = torch.where(states[:, 3] + self._params['speed_limit'] < 0)[0]
        if over_speed_agent.shape[0] > 0:
            a[over_speed_agent] = torch.clamp(a[over_speed_agent], min=0.)

        action = torch.cat([omega.unsqueeze(-1), a.unsqueeze(-1)], dim=-1)
        if torch.isnan(action).any():
            aaa = 0

        return action.reshape(-1, self.action_dim)

    def safe_mask(self, data: Union[Data, Batch], return_edge: bool = False) -> Tensor:
        if return_edge:
            pos_diff = data.edge_attr[:, :2]  # [i, j]: j -> i
            dist = pos_diff.norm(dim=-1)
            safe = torch.greater(dist, 4 * self._params['car_radius'])
            return safe

        if isinstance(data, Batch):
            data_list = data.to_data_list()
        else:
            data_list = [data]

        mask = []
        for graph in data_list:
            state_diff = graph.states.unsqueeze(1) - graph.states.unsqueeze(0)
            pos_diff = state_diff[graph.agent_mask, :, :2]  # [i, j]: j -> i
            dist = pos_diff.norm(dim=2)
            dist[:, :dist.shape[0]] += torch.eye(dist.shape[0], device=self.device) * (
                    4 * self._params['car_radius'] + 1)
            safe = torch.greater(dist, 3 * self._params['car_radius'])
            mask.append(torch.min(safe, dim=1)[0])
        mask = torch.cat(mask, dim=0).bool()

        return mask

    def unsafe_mask(self, data: Union[Data, Batch], return_edge: bool = False) -> Tensor:
        if return_edge:
            pos_diff = data.edge_attr[:, :2]  # [i, j]: j -> i
            dist = pos_diff.norm(dim=-1)
            collision = torch.less(dist, 2 * self._params['car_radius'])
            return collision

        warn_dist = 3 * self._params['car_radius']
        if isinstance(data, Batch):
            data_list = data.to_data_list()
        else:
            data_list = [data]
        mask = []
        for graph in data_list:
            # collision
            state_diff = graph.states.unsqueeze(1) - graph.states.unsqueeze(0)
            pos_diff = state_diff[graph.agent_mask, :, :2]  # [i, j]: j -> i
            dist = pos_diff.norm(dim=2)
            dist[:, :dist.shape[0]] += torch.eye(dist.shape[0], device=self.device) * (
                    4 * self._params['car_radius'] + 1)
            # dist = dist[graph.agent_mask, :]
            collision = torch.less(dist, 2 * self._params['car_radius'])
            graph_mask = torch.max(collision, dim=1)[0]

            # unsafe direction
            warn_zone = torch.less(dist, warn_dist)
            pos_vec = -(pos_diff / (torch.norm(pos_diff, dim=2, keepdim=True) + 0.0001))  # [i, j]: i -> j
            theta_state = graph.states[graph.agent_mask, 2].unsqueeze(1)
            theta_vec = torch.cat([torch.cos(theta_state), torch.sin(theta_state)], dim=1).repeat(
                pos_vec.shape[1], 1, 1).transpose(0, 1)  # [i, j]: theta[i]
            inner_prod = torch.sum(pos_vec * theta_vec, dim=2)
            unsafe_threshold = torch.cos(torch.asin(self._params['car_radius'] * 2 / (dist + 0.0000001)))
            unsafe = torch.greater(inner_prod, unsafe_threshold)
            unsafe = torch.max(torch.logical_and(unsafe, warn_zone), dim=1)[0]
            graph_mask = torch.logical_or(graph_mask, unsafe)

            mask.append(graph_mask)
        mask = torch.cat(mask, dim=0).bool()

        return mask

    def collision_mask(self, data: Union[Data, Batch]) -> Tensor:
        if isinstance(data, Batch):
            data_list = data.to_data_list()
        else:
            data_list = [data]

        if self._mode == 'train' or self._mode == 'test' or self._mode == 'demo_1' or self._mode == 'demo_2':
            mask = []
            for graph in data_list:
                state_diff = graph.states.unsqueeze(1) - graph.states.unsqueeze(0)
                pos_diff = state_diff[graph.agent_mask, :, :2]  # [i, j]: j -> i
                dist = pos_diff.norm(dim=2)
                dist[:, :dist.shape[0]] += torch.eye(dist.shape[0], device=self.device) * (
                        2 * self._params['car_radius'] + 1)
                collision = torch.less(dist, 2 * self._params['car_radius'])
                mask.append(torch.max(collision, dim=1)[0])
        elif self._mode == 'demo_0' or self._mode == 'demo_3':
            mask = []
            for graph in data_list:
                state_diff = graph.states[graph.agent_mask].unsqueeze(1) - graph.states[graph.agent_mask].unsqueeze(0)
                pos_diff = state_diff[:, :, :2]  # [i, j]: j -> i
                dist = pos_diff.norm(dim=2)
                dist += torch.eye(dist.shape[0], device=self.device) * (
                        2 * self._params['car_radius'] + 1)
                collision_agent = torch.less(dist, 2 * self._params['car_radius'])
                collision_agent = torch.max(collision_agent, dim=1)[0]
                collision_obs = torch.zeros_like(collision_agent)
                for i in range(len(self._agent_id)):
                    for j in range(len(self._obs_id)):
                        closest_points = p.getClosestPoints(
                            self._agent_id[i], self._obs_id[j],
                            0
                        )
                        if len(closest_points) > 0:
                            collision_obs[i] = 1
                mask.append(torch.logical_or(collision_agent, collision_obs))
        else:
            raise NotImplementedError
        mask = torch.cat(mask, dim=0).bool()
        return mask
