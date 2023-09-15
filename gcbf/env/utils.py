import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch

from scipy.linalg import inv, solve_discrete_are
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from matplotlib.pyplot import Axes
from typing import Optional
from torch import Tensor


def lqr(
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
):
    """
    Solve the discrete time lqr controller.
        x_{t+1} = A x_t + B u_t
        cost = sum x.T*Q*x + u.T*R*u
    Code adapted from Mark Wilfred Mueller's continuous LQR code at
    https://www.mwm.im/lqr-controllers-with-python/
    Based on Bertsekas, p.151
    Yields the control law u = -K x
    """

    # first, try to solve the Riccati equation
    X = solve_discrete_are(A, B, Q, R)

    # compute the LQR gain
    K = inv(B.T @ X @ B + R) @ (B.T @ X @ A)

    return K


def plot_graph(
        ax: Axes,
        data: Data,
        radius: float,
        color: str,
        with_label: bool = True,
        plot_edge: bool = False,
        alpha: float = 1.0,
        danger_radius: Optional[float] = None,
        safe_radius: Optional[float] = None,
        obstacle_color: str = '#000000',
) -> Axes:
    pos = data.pos.cpu().detach().numpy()

    def plot_node(i_node: int, node_color: str, node_label: bool = True, r: float = radius, a: float = alpha):
        ax.add_patch(plt.Circle((pos[i_node, 0], pos[i_node, 1]),
                                radius=r, color=node_color, clip_on=False, alpha=a))
        if node_label:
            ax.text(pos[i_node, 0], pos[i_node, 1], f'{i_node}', size=12, color="k",
                    family="sans-serif", weight="normal", horizontalalignment="center", verticalalignment="center",
                    transform=ax.transData, clip_on=True)
        if danger_radius is not None:
            ax.add_patch(
                plt.Circle((pos[i_node, 0], pos[i_node, 1]),
                           radius=danger_radius, color='red', clip_on=False, alpha=a, fill=False))
        if safe_radius is not None:
            ax.add_patch(
                plt.Circle((pos[i_node, 0], pos[i_node, 1]),
                           radius=safe_radius, color='green', clip_on=False, alpha=a, fill=False))

    for i in range(pos.shape[0]):
        if hasattr(data, 'agent_mask'):
            if data.agent_mask[i] == 1:
                plot_node(i, color, with_label)
            else:
                plot_node(i, obstacle_color, False, r=0.02, a=1)
        else:
            plot_node(i, color, with_label)
    if plot_edge:
        graph = to_networkx(data)
        nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5, node_size=500, arrowsize=10)
    return ax


def plot_node_3d(ax, pos: np.ndarray, r: float, color: str, alpha: float, grid: int = 10) -> Axes:
    u = np.linspace(0, 2 * np.pi, grid)
    v = np.linspace(0, np.pi, grid)
    x = r * np.outer(np.cos(u), np.sin(v)) + pos[0]
    y = r * np.outer(np.sin(u), np.sin(v)) + pos[1]
    z = r * np.outer(np.ones(np.size(u)), np.cos(v)) + pos[2]
    ax.plot_surface(x, y, z, color=color, alpha=alpha)
    return ax


def plot_graph_3d(
        ax,
        data: Data,
        radius: float,
        color: str,
        with_label: bool = True,
        plot_edge: bool = False,
        alpha: float = 1.0,
):
    pos = data.pos.cpu().detach().numpy()
    for i in range(pos.shape[0]):
        plot_node_3d(ax, pos[i], radius, color, alpha)
        if with_label:
            ax.text(pos[i, 0], pos[i, 1], pos[i, 2], f'{i}', size=12, color="k", family="sans-serif", weight="normal",
                    horizontalalignment="center", verticalalignment="center")
    if plot_edge:
        for i in range(data.edge_index[0].shape[0]):
            j, k = data.edge_index[0][i], data.edge_index[1][i]
            vec = pos[j, :] - pos[k, :]
            x = [pos[j, 0] - 2 * radius * vec[0], pos[k, 0] + 2 * radius * vec[0]]
            y = [pos[j, 1] - 2 * radius * vec[1], pos[k, 1] + 2 * radius * vec[1]]
            z = [pos[j, 2] - 2 * radius * vec[2], pos[k, 2] + 2 * radius * vec[2]]
            ax.plot(x, y, z, linewidth=1.0, color="k")
    return ax


def create_point_cloud_surface(vertices: Tensor, r: float) -> Tensor:
    points = []
    length = torch.norm(vertices[:, 1, :] - vertices[:, 0, :])
    width = torch.norm(vertices[:, 2, :] - vertices[:, 1, :])
    for i in range(1, int(length // (2 * r))):
        for j in range(int(width // (2 * r) + 1)):
            points.append(vertices[:, 0, :] + i * 2 * r * (vertices[:, 1, :] - vertices[:, 0, :]) / length +
                          j * 2 * r * (vertices[:, 2, :] - vertices[:, 1, :]) / width)
    for vertex in vertices:
        for i in range(4):
            points.append(vertex[i, :].unsqueeze(0))
    return torch.cat(points, dim=0)


def create_point_cloud(vertices: Tensor, r: float, dim: int = 2) -> Tensor:
    points = []
    if dim == 2:
        for i in range(vertices.shape[0]):
            points.append(vertices[i, :])
            j = i + 1 if i < vertices.shape[0] - 1 else 0
            direction = (vertices[j, :] - vertices[i, :]) / torch.norm(vertices[j, :] - vertices[i, :])
            while torch.norm(points[-1] - vertices[j, :]) > 2 * r:
                points.append(points[-1] + 2 * r * direction)
        points = torch.stack(points, dim=0)
    elif dim == 3:
        surface_nodes = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 4, 5, 1], [1, 2, 6, 5], [2, 6, 7, 3], [0, 3, 7, 4]]
        points = create_point_cloud_surface(vertices[surface_nodes, :], r)
    else:
        raise NotImplementedError
    return points


def create_rectangle(center: Tensor, length: float, width: float, theta: float) -> Tensor:
    vertices = torch.zeros((4, 2)).type_as(center)
    vertices[0, :] = torch.tensor([length / 2, width / 2]).type_as(center)
    vertices[1, :] = torch.tensor([length / 2, -width / 2]).type_as(center)
    vertices[2, :] = torch.tensor([-length / 2, -width / 2]).type_as(center)
    vertices[3, :] = torch.tensor([-length / 2, width / 2]).type_as(center)
    vertices = center + vertices @ torch.tensor([[np.cos(theta), -np.sin(theta)],
                                                 [np.sin(theta), np.cos(theta)]]).type_as(center)
    return vertices


def create_cuboid(center: Tensor, length: float, width: float, height: float, theta: float) -> Tensor:
    vertices = torch.zeros((8, 3)).type_as(center)
    vertices[0, :] = torch.tensor([length / 2, width / 2, height / 2]).type_as(center)
    vertices[1, :] = torch.tensor([length / 2, -width / 2, height / 2]).type_as(center)
    vertices[2, :] = torch.tensor([-length / 2, -width / 2, height / 2]).type_as(center)
    vertices[3, :] = torch.tensor([-length / 2, width / 2, height / 2]).type_as(center)
    vertices[4, :] = torch.tensor([length / 2, width / 2, -height / 2]).type_as(center)
    vertices[5, :] = torch.tensor([length / 2, -width / 2, -height / 2]).type_as(center)
    vertices[6, :] = torch.tensor([-length / 2, -width / 2, -height / 2]).type_as(center)
    vertices[7, :] = torch.tensor([-length / 2, width / 2, -height / 2]).type_as(center)
    vertices = center + vertices @ torch.tensor([[np.cos(theta), -np.sin(theta), 0],
                                                 [np.sin(theta), np.cos(theta), 0],
                                                 [0, 0, 1]]).type_as(center)
    return vertices
