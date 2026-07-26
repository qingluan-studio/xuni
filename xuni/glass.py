"""
XuniGlass —— 玻璃逻辑引擎

核心理念：计算过程应当像光穿过玻璃一样——透明、可追踪、可折射、可反射。

玻璃逻辑映射：
- 透明性 (Transparency)：每个计算步骤都留下"光迹"，完全可追溯
- 折射 (Refraction)：数据经过不同"介质"时改变路径和特性
- 反射 (Reflection)：输出能反馈回输入，形成共振回路
- 色散 (Dispersion)：不同"频率"的数据成分分离处理
- 聚焦 (Focusing)：能量汇聚到特定维度
- 全反射 (TIR)：超过临界角的数据完全反弹，形成隔离层

这不是日志系统，而是把"光学"作为软件架构本身。
每个函数都是一个"光学元件"，数据是"光"。
"""

import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
import inspect
import time


class OpticalMedium(Enum):
    """光学介质类型"""
    VACUUM = auto()      # 无变换直通
    GLASS = auto()       # 标准折射
    PRISM = auto()       # 色散分离
    MIRROR = auto()      # 反射
    LENS = auto()        # 聚焦/散焦
    FOG = auto()         # 散射/模糊
    CRYSTAL = auto()     # 双折射（多路径）


@dataclass
class LightRay:
    """
    光迹 —— 一段数据在玻璃逻辑中的完整旅行记录。

    不像传统日志只记录结果，光迹记录的是"路径本身"。
    """
    ray_id: str
    wavelength: float = 550.0  # nm，隐喻数据的"色调"/频段
    amplitude: float = 1.0
    phase: float = 0.0
    path: List[Tuple[str, float, float]] = field(default_factory=list)
    # path: [(element_name, entry_time, exit_time), ...]
    refractions: List[Dict] = field(default_factory=list)
    current_medium: OpticalMedium = OpticalMedium.VACUUM
    intensity: float = 1.0  # 经过损耗后的强度
    polarization: Tuple[float, float, float] = (0.0, 0.0, 1.0)  # 数据的"方向"
    payload: Any = None  # 实际携带的数据

    def enter(self, element_name: str):
        self.path.append((element_name, time.time(), None))

    def exit(self):
        if self.path:
            name, entry, _ = self.path[-1]
            self.path[-1] = (name, entry, time.time())

    def refract(self, from_medium: OpticalMedium, to_medium: OpticalMedium, angle: float):
        """记录折射事件"""
        self.refractions.append({
            "from": from_medium.name,
            "to": to_medium.name,
            "angle": angle,
            "timestamp": time.time(),
        })
        self.current_medium = to_medium

    def attenuate(self, factor: float):
        """衰减"""
        self.intensity *= factor

    def get_path_length(self) -> int:
        return len(self.path)

    def get_total_time(self) -> float:
        if not self.path:
            return 0.0
        start = self.path[0][1]
        end = self.path[-1][2] if self.path[-1][2] else time.time()
        return end - start


@dataclass
class OpticalElement:
    """
    光学元件 —— 一个具有光学特性的函数包装器。

    任何函数都可以被包装为光学元件，获得玻璃逻辑特性。
    """
    name: str
    medium: OpticalMedium
    refractive_index: float = 1.5  # 折射率：变换强度
    dispersion: float = 0.0  # 色散系数
    reflectivity: float = 0.0  # 反射率
    absorbance: float = 0.05  # 吸收率
    focus_power: float = 0.0  # 聚焦能力
    function: Optional[Callable] = None

    def process(self, ray: LightRay) -> LightRay:
        """处理光迹"""
        ray.enter(self.name)

        # 折射角（简化为折射率比）
        n1 = 1.0 if ray.current_medium == OpticalMedium.VACUUM else self.refractive_index
        n2 = self.refractive_index
        angle = np.arcsin(np.clip(n1 / n2 * np.sin(0.3), -1, 1)) if n2 != 0 else 0.0
        ray.refract(ray.current_medium, self.medium, float(angle))

        # 吸收损耗
        ray.attenuate(1.0 - self.absorbance)

        # 反射（如果反射率 > 0，产生分支光迹）
        reflected = None
        if self.reflectivity > 0 and ray.intensity > 0.1:
            reflected = LightRay(
                ray_id=ray.ray_id + "_R",
                wavelength=ray.wavelength,
                amplitude=ray.amplitude * self.reflectivity,
                intensity=ray.intensity * self.reflectivity,
                payload=ray.payload,
            )
            reflected.path = ray.path.copy()
            reflected.refractions = ray.refractions.copy()

        # 色散：对 payload 进行频段分离（如果 payload 是 ndarray）
        if self.medium == OpticalMedium.PRISM and isinstance(ray.payload, np.ndarray):
            # 模拟棱镜色散：分离"高频"和"低频"成分
            ray.payload = self._disperse(ray.payload)

        # 聚焦/散焦：改变数据的"集中度"
        if self.medium == OpticalMedium.LENS and isinstance(ray.payload, np.ndarray):
            ray.payload = self._focus(ray.payload)

        # 执行实际函数
        if self.function is not None:
            try:
                result = self.function(ray.payload)
                ray.payload = result
            except Exception as e:
                ray.payload = {"error": str(e), "original": ray.payload}
                ray.attenuate(0.5)

        ray.exit()
        return ray, reflected

    def _disperse(self, data: np.ndarray) -> np.ndarray:
        """棱镜色散：分离数据的极端值"""
        mean = np.mean(data)
        std = np.std(data) + 1e-12
        # "高频"（偏离均值）成分放大，"低频"压缩
        dispersed = mean + (data - mean) * (1 + self.dispersion)
        return dispersed

    def _focus(self, data: np.ndarray) -> np.ndarray:
        """透镜聚焦：增强中心趋势或极端值"""
        if self.focus_power == 0:
            return data
        # 正聚焦：向均值聚拢；负聚焦：向极端扩散
        mean = np.mean(data)
        focused = mean + (data - mean) * (1 - self.focus_power)
        return focused


class XuniGlass:
    """
    玻璃逻辑引擎。

    将计算 pipeline 建模为光学系统。
    数据 = 光，函数 = 光学元件，执行 = 光路追踪。
    """

    def __init__(self, name: str = "xuni_prism"):
        self.name = name
        self.elements: List[OpticalElement] = []
        self.rays: List[LightRay] = []
        self.reflected_rays: List[LightRay] = []
        self._ray_counter = 0

    def add_element(
        self,
        name: str,
        medium: OpticalMedium,
        func: Optional[Callable] = None,
        n: float = 1.5,
        dispersion: float = 0.0,
        reflectivity: float = 0.0,
        absorbance: float = 0.05,
        focus: float = 0.0,
    ):
        """添加光学元件到光路"""
        self.elements.append(OpticalElement(
            name=name,
            medium=medium,
            refractive_index=n,
            dispersion=dispersion,
            reflectivity=reflectivity,
            absorbance=absorbance,
            focus_power=focus,
            function=func,
        ))

    def shine(self, payload: Any, wavelength: float = 550.0) -> LightRay:
        """
        发射一束"光"（数据）进入光学系统。

        Returns:
            主光迹（透射光）
        """
        self._ray_counter += 1
        ray = LightRay(
            ray_id=f"{self.name}_{self._ray_counter}",
            wavelength=wavelength,
            payload=payload,
        )

        current_rays = [ray]
        for elem in self.elements:
            next_rays = []
            for r in current_rays:
                transmitted, reflected = elem.process(r)
                next_rays.append(transmitted)
                if reflected is not None:
                    self.reflected_rays.append(reflected)
            current_rays = next_rays

        # 取最强透射光作为主结果
        if current_rays:
            main_ray = max(current_rays, key=lambda r: r.intensity)
            self.rays.append(main_ray)
            return main_ray
        return ray

    def get_optical_report(self) -> dict:
        """
        获取完整的光学报告。
        这不是性能分析，而是"光路图"。
        """
        if not self.rays:
            return {}
        latest = self.rays[-1]
        return {
            "system_name": self.name,
            "elements": [e.name for e in self.elements],
            "total_rays": len(self.rays),
            "reflected_rays": len(self.reflected_rays),
            "latest_ray": {
                "id": latest.ray_id,
                "path_length": latest.get_path_length(),
                "travel_time_ms": latest.get_total_time() * 1000,
                "intensity": latest.intensity,
                "wavelength": latest.wavelength,
                "refractions": len(latest.refractions),
            },
            "path_trace": [
                {"element": name, "entry": entry, "exit": exit_t}
                for name, entry, exit_t in latest.path
            ],
        }

    def resonance_loop(self, payload: Any, iterations: int = 3, feedback_gain: float = 0.3) -> List[LightRay]:
        """
        反射共振：输出反馈回输入，形成光学共振腔。

        这模拟了激光腔的原理：光在两面镜子之间来回反射，
        每次增益放大，最终产生相干输出。
        """
        results = []
        current = payload
        for i in range(iterations):
            ray = self.shine(current, wavelength=400.0 + i * 100.0)
            results.append(ray)
            # 反馈：输出的一部分混合回输入
            if isinstance(ray.payload, np.ndarray):
                current = current * (1 - feedback_gain) + ray.payload * feedback_gain
            else:
                current = ray.payload
        return results
