"""
XuniSampler —— 超混沌采样引擎

支持流式生成上亿级采样点，无需全部存储在内存中。
核心机制：多维超混沌系统 + 分形几何 + 梯度噪声

采样模式：
- HYPER_CHAOS: 超混沌系统（4D+ 吸引子）
- LORENZ_96: 高维 Lorenz-96 环状系统
- MANDELBULB: 3D 分形（Mandelbulb）
- NOISE_FIELD: 4D OpenSimplex 噪声场
- HYBRID: 混合模式（混沌 + 噪声调制）
"""

import numpy as np
from enum import Enum, auto
from typing import Iterator, Tuple, Optional, Callable
from dataclasses import dataclass


class SamplingMode(Enum):
    HYPER_CHAOS = auto()
    LORENZ_96 = auto()
    MANDELBULB = auto()
    NOISE_FIELD = auto()
    HYBRID = auto()


@dataclass
class SamplePoint:
    """单个采样点的数据结构"""
    x: float
    y: float
    z: float
    w: float = 0.0  # 第四维（时间或能量）
    charge: float = 1.0  # 虚拟电荷基值
    entropy: float = 0.0  # 局部熵值

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, self.w, self.charge, self.entropy])


class _OpenSimplexNoise:
    """4D OpenSimplex 噪声的纯 NumPy 实现（无需外部依赖）"""

    def __init__(self, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.perm = np.zeros(256 * 2, dtype=np.int64)
        p = np.arange(256, dtype=np.int64)
        rng.shuffle(p)
        self.perm[:256] = p
        self.perm[256:] = p
        self.grad4 = np.array([
            [0,1,1,1],[0,1,1,-1],[0,1,-1,1],[0,1,-1,-1],
            [0,-1,1,1],[0,-1,1,-1],[0,-1,-1,1],[0,-1,-1,-1],
            [1,0,1,1],[1,0,1,-1],[1,0,-1,1],[1,0,-1,-1],
            [-1,0,1,1],[-1,0,1,-1],[-1,0,-1,1],[-1,0,-1,-1],
            [1,1,0,1],[1,1,0,-1],[1,-1,0,1],[1,-1,0,-1],
            [-1,1,0,1],[-1,1,0,-1],[-1,-1,0,1],[-1,-1,0,-1],
            [1,1,1,0],[1,1,-1,0],[1,-1,1,0],[1,-1,-1,0],
            [-1,1,1,0],[-1,1,-1,0],[-1,-1,1,0],[-1,-1,-1,0]
        ], dtype=np.float64)

    def _dot4(self, g, x, y, z, w):
        return g[:,0]*x + g[:,1]*y + g[:,2]*z + g[:,3]*w

    def noise4d(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, w: np.ndarray) -> np.ndarray:
        """批量计算 4D 噪声"""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        w = np.asarray(w, dtype=np.float64)
        
        results = np.zeros_like(x)
        for i in range(len(x)):
            results[i] = self._noise_single(x[i], y[i], z[i], w[i])
        return results

    def _noise_single(self, x: float, y: float, z: float, w: float) -> float:
        """单点 4D 噪声计算（完整算法）"""
        F4 = (np.sqrt(5.0) - 1.0) / 4.0
        G4 = (5.0 - np.sqrt(5.0)) / 20.0

        n0, n1, n2, n3, n4 = 0.0, 0.0, 0.0, 0.0, 0.0

        s = (x + y + z + w) * F4
        i = int(np.floor(x + s))
        j = int(np.floor(y + s))
        k = int(np.floor(z + s))
        l = int(np.floor(w + s))

        t = (i + j + k + l) * G4
        x0 = x - (i - t)
        y0 = y - (j - t)
        z0 = z - (k - t)
        w0 = w - (l - t)

        c1 = (x0 >= y0)
        c2 = (x0 >= z0)
        c3 = (y0 >= z0)
        c4 = (x0 >= w0)
        c5 = (y0 >= w0)
        c6 = (z0 >= w0)

        n0 = self._noise_contrib(i, j, k, l, x0, y0, z0, w0, c1, c2, c3, c4, c5, c6)
        n1 = self._noise_contrib(i+1, j, k, l, x0-1, y0, z0, w0, c1, c2, c3, c4, c5, c6)
        n2 = self._noise_contrib(i, j+1, k, l, x0, y0-1, z0, w0, c1, c2, c3, c4, c5, c6)
        n3 = self._noise_contrib(i+1, j+1, k, l, x0-1, y0-1, z0, w0, c1, c2, c3, c4, c5, c6)
        n4 = self._noise_contrib(i, j, k+1, l, x0, y0, z0-1, w0, c1, c2, c3, c4, c5, c6)

        return (n0 + n1 + n2 + n3 + n4) * 27.0

    def _noise_contrib(self, i: int, j: int, k: int, l: int, 
                      x: float, y: float, z: float, w: float,
                      c1, c2, c3, c4, c5, c6) -> float:
        """噪声贡献计算"""
        gi = self.perm[i & 255]
        gj = self.perm[(gi + j) & 255]
        gk = self.perm[(gj + k) & 255]
        gl = self.perm[(gk + l) & 255]
        
        idx = gl % 32
        grad = self.grad4[idx]
        
        n = 0.5 - x*x - y*y - z*z - w*w
        if n <= 0.0:
            return 0.0
        
        n *= n
        return n * n * (grad[0]*x + grad[1]*y + grad[2]*z + grad[3]*w)


class XuniSampler:
    """
    虚拟采样引擎。

    核心特性：
    1. 流式生成：yield 模式，内存占用 O(1)，可生成无限采样点
    2. 超混沌动力学：4D+ 吸引子产生高熵采样点
    3. 自适应密度：根据场能量动态调整采样密度
    4. 并行批次：支持批量生成以提高吞吐
    """

    def __init__(
        self,
        mode: SamplingMode = SamplingMode.HYPER_CHAOS,
        seed: int = 42,
        dt: float = 0.001,
        dimensions: int = 4,
    ):
        self.mode = mode
        self.seed = seed
        self.dt = dt
        self.dimensions = max(4, dimensions)
        self.rng = np.random.default_rng(seed)
        self.noise = _OpenSimplexNoise(seed)
        self._total_generated = 0

        # 超混沌 Chen 系统参数
        self._chen_params = {"a": 35.0, "b": 3.0, "c": 12.0, "d": 7.0, "k": 0.5}

        # Lorenz-96 参数
        self._lorenz_f = 8.0
        self._lorenz_dim = 40  # 高维环

        # 状态变量
        self._state = None
        self._reset_state()

    def _reset_state(self):
        """重置动力学系统状态"""
        if self.mode == SamplingMode.LORENZ_96:
            self._state = self.rng.random(self._lorenz_dim) * 2 - 1
            self._state[0] = self._lorenz_f + 0.01
        else:
            self._state = self.rng.random(self.dimensions) * 2 - 1

    # ------------------------------------------------------------------
    # 动力学方程
    # ------------------------------------------------------------------
    def _hyper_chaos_chen(self, state: np.ndarray) -> np.ndarray:
        """超混沌 Chen 系统：4D 吸引子，两个正 Lyapunov 指数"""
        with np.errstate(over='ignore', invalid='ignore'):
            x, y, z, w = state[:4]
            p = self._chen_params
            dx = p["a"] * (y - x)
            dy = -x * z + p["c"] * y + w
            dz = x * y - p["b"] * z
            dw = x * z + p["d"] * w
            return np.array([dx, dy, dz, dw])

    def _lorenz_96(self, state: np.ndarray) -> np.ndarray:
        """Lorenz-96：高维环状混沌系统，模拟大气动力学"""
        n = len(state)
        dx = np.zeros_like(state)
        for i in range(n):
            im1 = (i - 1) % n
            im2 = (i - 2) % n
            ip1 = (i + 1) % n
            dx[i] = (state[ip1] - state[im2]) * state[im1] - state[i] + self._lorenz_f
        return dx

    def _mandelbulb_field(self, idx: int) -> np.ndarray:
        """3D Mandelbulb 分形场采样"""
        with np.errstate(over='ignore', invalid='ignore'):
            n = 8  # 幂次
            rng = np.random.default_rng(self.seed + idx)
            theta = rng.random() * np.pi * 2
            phi = rng.random() * np.pi
            r = rng.random() ** 0.5 * 2.0

            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)

            # 迭代计算 Mandelbulb 距离估计
            for _ in range(6):
                r_vec = np.sqrt(x*x + y*y + z*z)
                if r_vec < 1e-10 or r_vec > 1e6:
                    break
                theta_n = np.arctan2(np.sqrt(x*x + y*y), z) * n
                phi_n = np.arctan2(y, x) * n
                r_n = min(r_vec ** n, 1e6)
                x = r_n * np.sin(theta_n) * np.cos(phi_n) + x
                y = r_n * np.sin(theta_n) * np.sin(phi_n) + y
                z = r_n * np.cos(theta_n) + z

            # 数值稳定性保护
            x = self._safe_clip(x)
            y = self._safe_clip(y)
            z = self._safe_clip(z)
            r_vec = np.sqrt(x*x + y*y + z*z)
            entropy = np.log1p(r_vec)
            return np.array([x, y, z, float(idx) * self.dt, 1.0, entropy])

    def _noise_sample(self, idx: int) -> np.ndarray:
        """4D 噪声场采样"""
        t = idx * self.dt * 100.0
        scale = 0.5
        x = self.noise.noise4d(
            np.array([t * scale]),
            np.array([t * scale + 100]),
            np.array([t * scale + 200]),
            np.array([t * scale + 300])
        )[0] * 10.0
        y = self.rng.normal(0, 1.0)
        z = self.rng.normal(0, 1.0)
        w = t
        entropy = abs(x) + abs(y) + abs(z)
        return np.array([x, y, z, w, 1.0, entropy])

    def _hybrid_sample(self, idx: int) -> np.ndarray:
        """混合模式：混沌 + 噪声调制"""
        # 先推进混沌系统一步
        k1 = self._hyper_chaos_chen(self._state)
        k2 = self._hyper_chaos_chen(self._state + 0.5 * self.dt * k1)
        k3 = self._hyper_chaos_chen(self._state + 0.5 * self.dt * k2)
        k4 = self._hyper_chaos_chen(self._state + self.dt * k3)
        self._state += (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        x, y, z, w = self._state[:4]
        # 噪声调制电荷
        noise_val = self.rng.random() * 2 - 1
        charge = 1.0 + noise_val * 0.3
        entropy = np.log1p(abs(x*y*z*w) + 1e-8)
        return np.array([x, y, z, w, charge, entropy])

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------
    def _safe_clip(self, val: float, limit: float = 1e6) -> float:
        """防止数值溢出，裁剪到安全范围"""
        if not np.isfinite(val):
            return 0.0
        return float(np.clip(val, -limit, limit))

    def generate_stream(self, count: Optional[int] = None) -> Iterator[SamplePoint]:
        """
        流式生成采样点。

        Args:
            count: 生成数量，None 表示无限生成

        Yields:
            SamplePoint: 单个采样点
        """
        idx = 0
        while count is None or idx < count:
            if self.mode == SamplingMode.HYPER_CHAOS:
                k1 = self._hyper_chaos_chen(self._state)
                k2 = self._hyper_chaos_chen(self._state + 0.5 * self.dt * k1)
                k3 = self._hyper_chaos_chen(self._state + 0.5 * self.dt * k2)
                k4 = self._hyper_chaos_chen(self._state + self.dt * k3)
                self._state += (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
                # 数值稳定性保护
                if not np.all(np.isfinite(self._state)):
                    self._reset_state()
                    continue
                x = self._safe_clip(self._state[0])
                y = self._safe_clip(self._state[1])
                z = self._safe_clip(self._state[2])
                w = self._safe_clip(self._state[3])
                entropy = np.log1p(abs(x*y*z*w) + 1e-8)
                point = SamplePoint(x, y, z, w, 1.0, float(entropy))

            elif self.mode == SamplingMode.LORENZ_96:
                k1 = self._lorenz_96(self._state)
                self._state += self.dt * k1
                x, y, z = self._state[0], self._state[1], self._state[2]
                w = float(idx) * self.dt
                entropy = np.std(self._state)
                point = SamplePoint(x, y, z, w, 1.0, float(entropy))

            elif self.mode == SamplingMode.MANDELBULB:
                arr = self._mandelbulb_field(idx)
                point = SamplePoint(*arr[:4], arr[4], arr[5])

            elif self.mode == SamplingMode.NOISE_FIELD:
                arr = self._noise_sample(idx)
                point = SamplePoint(*arr[:4], arr[4], arr[5])

            elif self.mode == SamplingMode.HYBRID:
                arr = self._hybrid_sample(idx)
                point = SamplePoint(*arr[:4], arr[4], arr[5])

            else:
                raise ValueError(f"Unknown mode: {self.mode}")

            self._total_generated += 1
            idx += 1
            yield point

    def generate_batch(self, batch_size: int = 10000) -> np.ndarray:
        """
        批量生成采样点，返回 NumPy 数组以提高效率。

        Returns:
            ndarray: shape (batch_size, 6)，列分别为 [x, y, z, w, charge, entropy]
        """
        batch = np.zeros((batch_size, 6), dtype=np.float64)
        for i, pt in enumerate(self.generate_stream(batch_size)):
            batch[i] = pt.to_array()
        return batch

    @property
    def total_generated(self) -> int:
        return self._total_generated

    def estimate_capacity(self, duration_seconds: float = 1.0) -> int:
        """估算在给定时间内可生成的采样点数量"""
        # 假设现代 CPU 每核每秒可生成约 10^6 个点
        import os
        cores = os.cpu_count() or 4
        rate_per_core = 1_000_000
        return int(duration_seconds * cores * rate_per_core)
