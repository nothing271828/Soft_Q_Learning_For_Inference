import torch
import torch.nn as nn


class QLayer(nn.Module):
    def __init__(self, x_dim, a_dim):
        """
        :param x_dim: 列向量x的长度
        :param a_dim: 列向量a的长度
        """
        super().__init__()
        # J: (x_dim, x_dim)
        self.J = nn.Parameter(torch.randn(x_dim, x_dim) * 0.1)
        # W: (x_dim, a_dim)
        self.W = nn.Parameter(torch.randn(x_dim, a_dim) * 0.1)
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.phi = torch.tanh
        self.x_dim = x_dim

    def forward(self, x, a):
        """
        :param x: batch * x_dim tensor
        :param a: batch * a_dim tensor
        :return: batch tensor
        """
        # φ(x) = tanh(x)
        phi_x = self.phi(x)
        phi_a = self.phi(a)

        term_J = phi_x @ self.J.T

        # term_W = W φ(a), 同理
        term_W = phi_a @ self.W.T

        residual = -x + term_J + term_W  # (batch,x_dim)

        q = - self.scale * (residual ** 2).sum(dim=1) / self.x_dim + self.bias

        return q


class QLayer2(nn.Module):
    """
    另一种网络结构，x通过一个网络连接到J_{ij}，再理解其作为连接矩阵计算
    """

    def __init__(self, x_dim, a_dim):
        """
        :param x_dim: 列向量x的长度
        :param a_dim: 列向量a的长度
        """
        super().__init__()
        # J: (x_dim, x_dim)
        self.j_generator = nn.Sequential(
            nn.Linear(x_dim, a_dim * a_dim),
            nn.Tanh(),
            nn.Linear(a_dim * a_dim, a_dim * a_dim),
            nn.Tanh(),
            nn.Linear(a_dim * a_dim, a_dim * a_dim),
            nn.Tanh()
        )
        self.phi = nn.Tanh()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.x_dim = x_dim
        self.a_dim = a_dim

    def forward(self, x, a):
        """
        :param x: batch * x_dim tensor
        :param a: batch * a_dim tensor
        :return: batch tensor
        """
        x = x.float()
        a = a.float()
        j_flat = self.j_generator(x)
        J = j_flat.view(x.size(0), self.a_dim, self.a_dim)
        phi_a = self.phi(a)
        interaction_term = torch.einsum('bij,bj->bi', J, phi_a)

        residual = -a + interaction_term
        q = - self.scale * (residual ** 2).sum(dim=1) / self.x_dim + self.bias

        return q
