"""
XuniHydro —— 水动力学采样引擎

核心理念：采样点不是离散的数字，而是"流体粒子"。
它们有质量、速度、压力、温度，在虚拟空间中流动、汇聚、蒸发、凝结。

水逻辑映射：
- 流动性：粒子服从简化 Navier-Stokes 方程
- 汇聚：高密度区域形成"深潭"，产生低频共鸣
- 蒸发：高能粒子脱离液态，转化为"气态"能量输入电场
- 凝结：低能区域自发产生新粒子
- 涡旋：旋度产生音乐中的颤音与和声缠绕
- 表面张力：粒子间吸引/排斥的微妙平衡

这不是传统粒子系统，而是把"水"作为计算隐喻本身。
"""

import numpy as np
from typing import Iterator, List, Tuple
from dataclasses import dataclass, field


@dataclass
class FluidParticle:
    """
    流体粒子 —— 一滴"虚拟水"的完整状态。
    """
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    mass: float = 1.0
    density: float = 1.0
    pressure: float = 0.0
    temperature: float = 300.0  # Kelvin (virtual)
    phase: str = "liquid"  # liquid | gas | plasma
    age: int = 0
    vorticity: float = 0.0

    def kinetic_energy(self) -> float:
        v2 = self.vx**2 + self.vy**2 + self.vz**2
        return 0.5 * self.mass * v2

    def to_array(self) -> np.ndarray:
        return np.array([
            self.x, self.y, self.z,
            self.vx, self.vy, self.vz,
            self.mass, self.density, self.pressure,
            self.temperature, self.vorticity
        ])


class _SPHKernel:
    """
    简化 SPH（光滑粒子流体动力学）核函数。
    不需要邻居搜索树，使用全局场近似。
    """

    def __init__(self, smoothing_length: float = 2.0):
        self.h = smoothing_length
        self.h2 = smoothing_length ** 2
        self.h3 = smoothing_length ** 3

    def poly6(self, r2: float) -> float:
        """Poly6 核，用于密度估计"""
        if r2 >= self.h2:
            return 0.0
        return (315.0 / (64.0 * np.pi * self.h3)) * (self.h2 - r2) ** 3

    def spiky_gradient(self, r: float) -> float:
        """Spiky 核梯度，用于压力计算"""
        if r >= self.h or r < 1e-8:
            return 0.0
        return (-45.0 / (np.pi * self.h3)) * (self.h - r) ** 2

    def viscosity_laplacian(self, r: float) -> float:
        """Viscosity 核拉普拉斯，用于粘性"""
        if r >= self.h:
            return 0.0
        return (45.0 / (np.pi * self.h3)) * (self.h - r)


class XuniHydro:
    """
    水动力学采样引擎。

    用流体物理来驱动采样点的生成与演化。
    核心方程（简化 Navier-Stokes）：
        dv/dt = -∇p/ρ + ν∇²v + f_external + f_vorticity
    """

    def __init__(
        self,
        n_particles: int = 4096,
        dt: float = 0.005,
        gravity: Tuple[float, float, float] = (0.0, -0.5, 0.0),
        viscosity: float = 0.1,
        vorticity_strength: float = 0.05,
        evap_threshold: float = 800.0,
        condense_threshold: float = 200.0,
        bounds: Tuple[float, float] = (-20.0, 20.0),
        seed: int = 42,
    ):
        self.n_particles = n_particles
        self.dt = dt
        self.gravity = np.array(gravity, dtype=np.float64)
        self.viscosity = viscosity
        self.vorticity_strength = vorticity_strength
        self.evap_threshold = evap_threshold  # 温度超过此值蒸发
        self.condense_threshold = condense_threshold  # 温度低于此值凝结
        self.bounds = bounds
        self.rng = np.random.default_rng(seed)
        self.kernel = _SPHKernel(smoothing_length=3.0)

        # 初始化粒子
        self.particles: List[FluidParticle] = []
        self._init_particles()

        # 场网格（用于快速密度估计）
        self.grid_size = 16
        self._grid = np.zeros((self.grid_size, self.grid_size, self.grid_size))

    def _init_particles(self):
        """初始化粒子云 —— 像一滴水落入容器"""
        for _ in range(self.n_particles):
            p = FluidParticle(
                x=self.rng.normal(0, 3.0),
                y=self.rng.normal(0, 3.0),
                z=self.rng.normal(0, 3.0),
                vx=self.rng.normal(0, 0.5),
                vy=self.rng.normal(0, 0.5),
                vz=self.rng.normal(0, 0.5),
                mass=self.rng.uniform(0.8, 1.2),
                temperature=self.rng.uniform(250, 350),
            )
            self.particles.append(p)

    def _update_grid(self):
        """将粒子投影到粗略网格，用于快速密度估计"""
        self._grid.fill(0.0)
        gs = self.grid_size
        bmin, bmax = self.bounds
        scale = gs / (bmax - bmin)
        for p in self.particles:
            ix = int((p.x - bmin) * scale)
            iy = int((p.y - bmin) * scale)
            iz = int((p.z - bmin) * scale)
            if 0 <= ix < gs and 0 <= iy < gs and 0 <= iz < gs:
                self._grid[ix, iy, iz] += p.mass

    def _compute_density_pressure(self, p: FluidParticle) -> Tuple[float, float]:
        """使用网格近似计算局部密度和压力"""
        gs = self.grid_size
        bmin, bmax = self.bounds
        scale = gs / (bmax - bmin)
        ix = int((p.x - bmin) * scale)
        iy = int((p.y - bmin) * scale)
        iz = int((p.z - bmin) * scale)

        density = 0.0
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    gx, gy, gz = ix+dx, iy+dy, iz+dz
                    if 0 <= gx < gs and 0 <= gy < gs and 0 <= gz < gs:
                        cell_mass = self._grid[gx, gy, gz]
                        cx = bmin + (gx + 0.5) / scale
                        cy = bmin + (gy + 0.5) / scale
                        cz = bmin + (gz + 0.5) / scale
                        r2 = (p.x-cx)**2 + (p.y-cy)**2 + (p.z-cz)**2
                        density += cell_mass * self.kernel.poly6(r2)

        # 状态方程：p = k(ρ - ρ₀)
        pressure = max(0.0, 0.5 * (density - 1.0))
        return density, pressure

    def _compute_vorticity(self, p: FluidParticle) -> np.ndarray:
        """计算涡旋力（旋度近似）"""
        gs = self.grid_size
        bmin, bmax = self.bounds
        scale = gs / (bmax - bmin)
        ix = int((p.x - bmin) * scale)
        iy = int((p.y - bmin) * scale)
        iz = int((p.z - bmin) * scale)

        # 用相邻网格速度差近似旋度
        curl = np.array([0.0, 0.0, 0.0])
        if 0 < ix < gs-1 and 0 < iy < gs-1 and 0 < iz < gs-1:
            # 简化的旋度计算（仅作方向性涡旋力）
            curl[0] = (self._grid[ix, iy+1, iz] - self._grid[ix, iy-1, iz])
            curl[1] = -(self._grid[ix+1, iy, iz] - self._grid[ix-1, iy, iz])
            curl[2] = (self._grid[ix, iy, iz+1] - self._grid[ix, iy, iz-1])

        # 涡旋力 = ε(N × ω)，其中 N 是旋度方向归一化
        omega = np.array([p.vx, p.vy, p.vz])
        norm_curl = np.linalg.norm(curl) + 1e-8
        force = self.vorticity_strength * np.cross(curl / norm_curl, omega)
        return force

    def _step(self):
        """单步流体演化"""
        self._update_grid()

        new_particles = []
        evaporated_energy = 0.0

        for p in self.particles:
            p.age += 1

            # 计算密度和压力
            density, pressure = self._compute_density_pressure(p)
            p.density = density
            p.pressure = pressure

            # 压力梯度力（简化为指向低密度方向的力）
            if density > 1.5:
                # 过密 → 排斥
                pressure_force = -0.1 * (density - 1.5) * np.array([p.x, p.y, p.z])
            else:
                pressure_force = np.array([0.0, 0.0, 0.0])

            # 粘性力（与速度反向）
            velocity = np.array([p.vx, p.vy, p.vz])
            viscous_force = -self.viscosity * velocity

            # 涡旋力
            vorticity_force = self._compute_vorticity(p)

            # 总力
            total_force = pressure_force + viscous_force + vorticity_force + self.gravity * p.mass
            acceleration = total_force / p.mass

            # 更新速度和位置（半隐式欧拉）
            velocity += acceleration * self.dt
            p.vx, p.vy, p.vz = velocity[0], velocity[1], velocity[2]
            p.x += p.vx * self.dt
            p.y += p.vy * self.dt
            p.z += p.vz * self.dt

            # 边界反弹（像水碰到玻璃壁）
            bmin, bmax = self.bounds
            damp = 0.6
            for coord in ['x', 'y', 'z']:
                val = getattr(p, coord)
                vcoord = 'v' + coord
                vval = getattr(p, vcoord)
                if val < bmin:
                    setattr(p, coord, bmin + (bmin - val))
                    setattr(p, vcoord, -vval * damp)
                elif val > bmax:
                    setattr(p, coord, bmax - (val - bmax))
                    setattr(p, vcoord, -vval * damp)

            # 温度更新：动能转化为热 + 环境热交换
            ke = p.kinetic_energy()
            p.temperature = 300.0 + ke * 50.0

            # 相变：蒸发
            if p.temperature > self.evap_threshold:
                evaporated_energy += ke
                continue  # 粒子蒸发，不加入新列表

            # 相变：凝结标记（用于后续在低密度区域生成新粒子）
            if p.temperature < self.condense_threshold and density < 0.5:
                p.phase = "condensing"
            else:
                p.phase = "liquid"

            # 涡旋强度记录
            p.vorticity = np.linalg.norm(vorticity_force)
            new_particles.append(p)

        # 凝结：在低密度高温区域生成新粒子（能量守恒）
        if evaporated_energy > 10.0 and len(new_particles) < self.n_particles * 2:
            n_condense = min(10, int(evaporated_energy / 5.0))
            for _ in range(n_condense):
                # 在随机低密度区域凝结
                p = FluidParticle(
                    x=self.rng.uniform(*self.bounds),
                    y=self.rng.uniform(*self.bounds),
                    z=self.rng.uniform(*self.bounds),
                    vx=self.rng.normal(0, 0.2),
                    vy=self.rng.normal(0, 0.2),
                    vz=self.rng.normal(0, 0.2),
                    temperature=self.condense_threshold * 0.8,
                    phase="liquid",
                )
                new_particles.append(p)

        self.particles = new_particles
        return evaporated_energy

    def generate_stream(self, steps: int = 1000) -> Iterator[FluidParticle]:
        """
        流式生成流体粒子状态。
        每步演化后 yield 当前所有粒子。
        """
        for _ in range(steps):
            self._step()
            for p in self.particles:
                yield p

    def get_sample_batch(self, n: int = 10000) -> np.ndarray:
        """
        将当前流体状态转换为与 XuniSampler 兼容的采样点格式。
        返回 shape (N, 6): [x, y, z, w, charge, entropy]
        其中 w=温度, charge=密度, entropy=涡旋强度
        """
        # 先演化几步达到稳态
        for _ in range(50):
            self._step()

        batch = np.zeros((n, 6))
        idx = 0
        while idx < n:
            self._step()
            for p in self.particles:
                if idx >= n:
                    break
                batch[idx] = [
                    p.x, p.y, p.z,
                    p.temperature / 1000.0,  # w = 归一化温度
                    p.density,  # charge = 密度
                    p.vorticity,  # entropy = 涡旋
                ]
                idx += 1
        return batch

    def hydro_summary(self) -> dict:
        """流体状态摘要"""
        if not self.particles:
            return {
                "particle_count": 0,
                "temperature_mean": 0.0,
                "temperature_max": 0.0,
                "density_mean": 0.0,
                "vorticity_mean": 0.0,
                "phases": {},
            }
        temps = [p.temperature for p in self.particles]
        densities = [p.density for p in self.particles]
        vorticities = [p.vorticity for p in self.particles]
        phases = {}
        for p in self.particles:
            phases[p.phase] = phases.get(p.phase, 0) + 1
        return {
            "particle_count": len(self.particles),
            "temperature_mean": float(np.mean(temps)),
            "temperature_max": float(np.max(temps)),
            "density_mean": float(np.mean(densities)),
            "vorticity_mean": float(np.mean(vorticities)),
            "phases": phases,
        }
