"""
XuniField —— 虚拟电场系统

将采样点的空间分布转换为虚拟电荷，进而构建电势场与电场。
数学基础：
- 电荷密度 ρ(x) 与采样点密度成正比
- 电势 V 满足泊松方程近似：∇²V ≈ -ρ/ε₀（离散化求解）
- 电场 E = -∇V
"""

import numpy as np
from typing import Iterator, Optional, Tuple, List
from dataclasses import dataclass
from .sampler import SamplePoint


@dataclass
class FieldCell:
    """场网格单元"""
    i: int
    j: int
    k: int
    density: float       # 采样点密度
    charge: float        # 虚拟电荷量
    potential: float     # 电势 V
    ex: float            # 电场 x 分量
    ey: float            # 电场 y 分量
    ez: float            # 电场 z 分量
    energy: float        # 能量密度 = 0.5 * ε * |E|²


class XuniField:
    """
    虚拟电场。

    通过将采样点投影到离散网格上，计算局部密度→电荷→电势→电场。
    支持增量更新：新采样点到达时动态更新场分布。
    """

    def __init__(
        self,
        grid_size: Tuple[int, int, int] = (32, 32, 32),
        bounds: Tuple[float, float] = (-50.0, 50.0),
        epsilon: float = 1.0,
        smooth_sigma: float = 1.5,
    ):
        self.nx, self.ny, self.nz = grid_size
        self.xmin, self.xmax = bounds
        self.ymin, self.ymax = bounds
        self.zmin, self.zmax = bounds
        self.epsilon = epsilon  # 介电常数（虚拟）
        self.smooth_sigma = smooth_sigma

        # 网格间距
        self.dx = (self.xmax - self.xmin) / self.nx
        self.dy = (self.ymax - self.ymin) / self.ny
        self.dz = (self.zmax - self.zmin) / self.nz

        # 场数组
        self.density = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64)
        self.charge = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64)
        self.potential = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64)
        self.ex = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64)
        self.ey = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64)
        self.ez = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64)
        self.energy = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64)

        self._total_samples = 0
        self._field_ready = False

    def reset(self):
        """重置所有场数组，准备新一轮计算"""
        self.density = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64)
        self.charge = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64)
        self.potential = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64)
        self.ex = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64)
        self.ey = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64)
        self.ez = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64)
        self.energy = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64)
        self._total_samples = 0
        self._field_ready = False

    def _world_to_grid(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        """世界坐标转网格索引"""
        ix = int((x - self.xmin) / self.dx)
        iy = int((y - self.ymin) / self.dy)
        iz = int((z - self.zmin) / self.dz)
        return (
            max(0, min(self.nx - 1, ix)),
            max(0, min(self.ny - 1, iy)),
            max(0, min(self.nz - 1, iz)),
        )

    def ingest_stream(self, points: Iterator[SamplePoint], count: int = 10000):
        """
        吸收采样点流，更新密度场。

        Args:
            points: 采样点迭代器
            count: 本次吸收的数量
        """
        for _ in range(count):
            try:
                pt = next(points)
            except StopIteration:
                break
            ix, iy, iz = self._world_to_grid(pt.x, pt.y, pt.z)
            self.density[ix, iy, iz] += pt.charge
            self._total_samples += 1

        self._field_ready = False

    def ingest_batch(self, batch: np.ndarray):
        """
        批量吸收采样点（NumPy 数组，shape (N, 6)）。
        """
        xs, ys, zs = batch[:, 0], batch[:, 1], batch[:, 2]
        charges = batch[:, 4]

        ixs = ((xs - self.xmin) / self.dx).astype(np.int64)
        iys = ((ys - self.ymin) / self.dy).astype(np.int64)
        izs = ((zs - self.zmin) / self.dz).astype(np.int64)

        # 裁剪到合法范围
        ixs = np.clip(ixs, 0, self.nx - 1)
        iys = np.clip(iys, 0, self.ny - 1)
        izs = np.clip(izs, 0, self.nz - 1)

        for ix, iy, iz, q in zip(ixs, iys, izs, charges):
            self.density[ix, iy, iz] += q

        self._total_samples += len(batch)
        self._field_ready = False

    def compute_field(self, iterations: int = 100):
        """
        计算电荷、电势和电场。

        使用离散泊松方程的松弛法求解：
        V_new[i,j,k] = (V[i+1] + V[i-1] + V[j+1] + V[j-1] + V[k+1] + V[k-1] + ρ*dx²/ε) / 6
        """
        if self._total_samples == 0:
            return

        # 1. 密度 → 电荷（非线性映射，避免极端值）
        rho = np.tanh(self.density * 0.1)  # 饱和函数
        self.charge = rho

        # 2. 可选：高斯平滑
        if self.smooth_sigma > 0:
            from scipy.ndimage import gaussian_filter
            rho = gaussian_filter(rho, sigma=self.smooth_sigma)

        # 3. 求解电势（Jacobi 松弛）
        V = self.potential.copy()
        dx2 = self.dx ** 2
        factor = dx2 / self.epsilon

        for _ in range(iterations):
            V_new = V.copy()
            # 内部点更新
            V_new[1:-1, 1:-1, 1:-1] = (
                V[2:, 1:-1, 1:-1] + V[:-2, 1:-1, 1:-1] +
                V[1:-1, 2:, 1:-1] + V[1:-1, :-2, 1:-1] +
                V[1:-1, 1:-1, 2:] + V[1:-1, 1:-1, :-2] +
                factor * rho[1:-1, 1:-1, 1:-1]
            ) / 6.0
            V = V_new

        self.potential = V

        # 4. 计算电场 E = -∇V（中心差分）
        self.ex[1:-1, 1:-1, 1:-1] = -(V[2:, 1:-1, 1:-1] - V[:-2, 1:-1, 1:-1]) / (2 * self.dx)
        self.ey[1:-1, 1:-1, 1:-1] = -(V[1:-1, 2:, 1:-1] - V[1:-1, :-2, 1:-1]) / (2 * self.dy)
        self.ez[1:-1, 1:-1, 1:-1] = -(V[1:-1, 1:-1, 2:] - V[1:-1, 1:-1, :-2]) / (2 * self.dz)

        # 5. 能量密度 u = 0.5 * ε * |E|²
        e2 = self.ex**2 + self.ey**2 + self.ez**2
        self.energy = 0.5 * self.epsilon * e2

        self._field_ready = True

    def get_total_energy(self) -> float:
        """获取场的总能量"""
        if not self._field_ready:
            self.compute_field()
        return float(np.sum(self.energy))

    def get_energy_distribution(self) -> np.ndarray:
        """获取能量分布（展平数组）"""
        if not self._field_ready:
            self.compute_field()
        return self.energy.flatten()

    def get_dominant_vector(self) -> Tuple[float, float, float]:
        """获取主导电场方向（能量加权平均）"""
        if not self._field_ready:
            self.compute_field()
        total_e = np.sum(self.energy) + 1e-12
        wx = np.sum(self.ex * self.energy) / total_e
        wy = np.sum(self.ey * self.energy) / total_e
        wz = np.sum(self.ez * self.energy) / total_e
        return (float(wx), float(wy), float(wz))

    def get_cells(self, threshold: float = 0.01) -> List[FieldCell]:
        """获取能量超过阈值的场单元列表"""
        if not self._field_ready:
            self.compute_field()
        cells = []
        emax = np.max(self.energy) + 1e-12
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    if self.energy[i, j, k] / emax > threshold:
                        cells.append(FieldCell(
                            i=i, j=j, k=k,
                            density=float(self.density[i,j,k]),
                            charge=float(self.charge[i,j,k]),
                            potential=float(self.potential[i,j,k]),
                            ex=float(self.ex[i,j,k]),
                            ey=float(self.ey[i,j,k]),
                            ez=float(self.ez[i,j,k]),
                            energy=float(self.energy[i,j,k]),
                        ))
        return cells

    def field_summary(self) -> dict:
        """场的摘要统计"""
        if not self._field_ready:
            self.compute_field()
        return {
            "total_samples": self._total_samples,
            "grid_size": [self.nx, self.ny, self.nz],
            "bounds": [self.xmin, self.xmax],
            "total_energy": float(np.sum(self.energy)),
            "max_potential": float(np.max(self.potential)),
            "min_potential": float(np.min(self.potential)),
            "max_field_strength": float(np.max(np.sqrt(self.ex**2 + self.ey**2 + self.ez**2))),
            "energy_mean": float(np.mean(self.energy)),
            "energy_std": float(np.std(self.energy)),
        }
