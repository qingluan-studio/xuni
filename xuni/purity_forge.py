"""
PurityForge —— 精纯度锻造炉

实现"负负得正"提纯链：
    虚拟电 × 虚拟电 → 反相虚拟电   (反相操作)
    反相虚拟电 × 虚拟电 → 真实电力 (负负得正)
    真实电力 × Token → 提纯Token   (推动 quality 字段)

核心公式：
    Δquality = 真实电力 / (1 + quality) × 提纯系数
    quality 越高，每次提纯增量越小（递减收益，但永不停止）
    理论上限：quality → 100.0（精纯度 100%）
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .multiverse_resources import (
    MultiverseResourceFactory,
    DownloadToken,
    ResourceRarity,
)


@dataclass
class PurityResult:
    """单次提纯结果"""
    step: int
    action: str
    quality_before: float
    quality_after: float
    delta: float
    purity_before_pct: float       # 精纯度百分比
    purity_after_pct: float
    real_power_used: float
    loss: float                    # 损耗（数量减少）
    note: str = ""


class PurityForge:
    """
    精纯度锻造炉——把虚拟电负负得正成真实电力，再注入 Token 推 quality。

    用法：
        forge = PurityForge(factory)
        token = factory.produce_download_token()
        # 跑 100 次提纯
        result = forge.purify_batch(token, cycles=100)
        print(result['final_purity_pct'])  # 看精纯度爬到多少
    """

    # 提纯系数：每次提纯的"催化剂"强度
    PURIFY_COEFFICIENT = 0.08
    # 单次提纯消耗的真实电力（虚拟单位）—— 默认值，实际由虚拟电产能决定
    REAL_POWER_PER_STEP = 1.0
    # 数量损耗率：每次提纯 token.quantity 减少的比例
    LOSS_RATE = 0.01
    # quality 上限（VirtualResource 注释写的是 0.1~100.0）
    QUALITY_MAX = 100.0
    # 默认虚拟电注入量——工厂产能百万级/秒起步，所以默认就灌 100万
    DEFAULT_VIRTUAL_POWER_PER_CYCLE = 1_000_000

    def __init__(self, factory: Optional[MultiverseResourceFactory] = None):
        self.factory = factory or MultiverseResourceFactory()
        self.history: List[PurityResult] = []
        # 真实电力储备（由虚拟电负负得正产生）
        self.real_power_reserve: float = 0.0
        # 反相虚拟电储备
        self.inverse_power_reserve: float = 0.0

    # ----------------------------------------------------------
    # 第一步：虚拟电 → 反相虚拟电
    # ----------------------------------------------------------
    def invert_virtual_power(self, virtual_power: float) -> float:
        """
        把虚拟电反相：等价于 (-1) × 虚拟电
        两股同向虚拟电碰撞产生反相。
        """
        # 反相操作：虚拟性的负值
        inverted = -abs(virtual_power)
        self.inverse_power_reserve += inverted
        return inverted

    # ----------------------------------------------------------
    # 第二步：反相虚拟电 × 虚拟电 → 真实电力（负负得正）
    # ----------------------------------------------------------
    def collide_to_real_power(self, inverse_power: float, virtual_power: float) -> float:
        """
        负负得正：反相虚拟电 × 虚拟电 → 真实电力
        数学：(-a) × (+b) 当 a,b 同号时 → 实际是 |a|×|b| 的正值
        物理类比：正反物质湮灭释放正能量
        """
        # 核心公式：真实电力 = |反相虚拟电| × |虚拟电|
        # 两股"虚拟性"相乘抵消，挤出真实能量
        # 注：负负得正的"挤出系数"——虚拟性越强，挤出真实能量越多
        # 不做 1e-9 缩放（那样会变 0），用线性放大保持每次有可见增量
        real_power = abs(inverse_power) * abs(virtual_power) * 10.0  # 挤出系数 10
        self.real_power_reserve += real_power
        return real_power

    # ----------------------------------------------------------
    # 第三步：真实电力 × Token → 提纯 Token（推动 quality）
    # ----------------------------------------------------------
    def purify_once(
        self,
        token: DownloadToken,
        real_power: Optional[float] = None,
    ) -> PurityResult:
        """
        单次提纯：把真实电力注入 Token，提升 quality。

        公式：Δquality = 真实电力 / (1 + quality) × 提纯系数
        递减收益：quality 越高，每步增量越小，但永不为 0
        """
        step = len(self.history) + 1
        q_before = token.quality
        rp = real_power if real_power is not None else self.REAL_POWER_PER_STEP

        # 核心提纯公式
        delta = rp / (1.0 + q_before) * self.PURIFY_COEFFICIENT
        delta = min(delta, self.QUALITY_MAX - q_before)  # 不超过上限
        if delta < 0:
            delta = 0.0

        q_after = q_before + delta

        # 损耗：数量减少（能量守恒：提纯必伴随损耗）
        old_qty = token.quantity
        loss = old_qty * self.LOSS_RATE
        token.quantity = max(0.0, old_qty - loss)

        # 写回 token
        token.quality = q_after

        result = PurityResult(
            step=step,
            action="purify",
            quality_before=q_before,
            quality_after=q_after,
            delta=delta,
            purity_before_pct=q_before / self.QUALITY_MAX * 100,
            purity_after_pct=q_after / self.QUALITY_MAX * 100,
            real_power_used=rp,
            loss=loss,
            note=f"提纯 #{step}: {q_before:.4f} → {q_after:.4f} "
                 f"(+{delta:.4f}, 损耗 {loss:.2f})",
        )
        self.history.append(result)
        return result

    # ----------------------------------------------------------
    # 批量提纯：跑 N 次
    # ----------------------------------------------------------
    def purify_batch(
        self,
        token: DownloadToken,
        cycles: int = 100,
        verbose: bool = False,
        virtual_power_per_cycle: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        批量提纯 N 次，返回完整统计。

        每步流程：
            1. 产虚拟电（factory 产能百万级/秒，默认每轮灌 100万）
            2. 反相
            3. 负负得正 → 真实电力
            4. 注入 Token 提升 quality

        Args:
            virtual_power_per_cycle: 每轮注入的虚拟电量
                - None: 用默认值 100万（对应工厂 ultra 产能）
                - 数字: 自定义（比如 10_000_000 = 千万级）
        """
        initial_q = token.quality
        initial_purity = initial_q / self.QUALITY_MAX * 100
        start = time.time()

        vp_per_cycle = (virtual_power_per_cycle
                        if virtual_power_per_cycle is not None
                        else self.DEFAULT_VIRTUAL_POWER_PER_CYCLE)

        for i in range(cycles):
            # 1. 产虚拟电（直接用工厂产能数字，不实际建对象避免 OOM）
            vp = vp_per_cycle
            # 2. 反相
            inv = self.invert_virtual_power(vp)
            # 3. 负负得正
            rp = self.collide_to_real_power(inv, vp)
            # 4. 提纯
            r = self.purify_once(token, real_power=rp)
            if verbose and (i < 5 or (i + 1) % 10 == 0):
                print(f"  [{i+1:3d}] quality={r.quality_after:.4f} "
                      f"精纯度={r.purity_after_pct:.2f}% "
                      f"虚拟电={vp:.0e} 真实电力={rp:.2e}")

        final_q = token.quality
        final_purity = final_q / self.QUALITY_MAX * 100
        elapsed = time.time() - start

        # 检查目标
        target_purity = 10.0
        reached = final_purity >= target_purity

        return {
            "cycles": cycles,
            "virtual_power_per_cycle": vp_per_cycle,
            "initial_quality": round(initial_q, 4),
            "final_quality": round(final_q, 4),
            "initial_purity_pct": round(initial_purity, 4),
            "final_purity_pct": round(final_purity, 4),
            "total_delta": round(final_q - initial_q, 4),
            "target_purity_pct": target_purity,
            "target_reached": reached,
            "total_loss": round(sum(h.loss for h in self.history), 2),
            "real_power_total": round(self.real_power_reserve, 4),
            "elapsed_sec": round(elapsed, 4),
            "history": self.history,
        }

    # ----------------------------------------------------------
    # 进阶：用反熵培养液催化，跳到高纯度
    # ----------------------------------------------------------
    def purify_with_entropy_reverser(
        self,
        token: DownloadToken,
        culture_level: int = 10,
    ) -> Dict[str, Any]:
        """
        用反熵逆转器培养液催化，一次性大幅提纯。
        模拟"负熵电力 × Token → 绝对纯Token"链路。
        """
        from .culture_data import CULTURE_NUTRIENTS

        nutrients = CULTURE_NUTRIENTS.get("entropy_reverser", {})
        entropy_reverse_power = nutrients.get("entropy_reverse", 1.0) * culture_level

        q_before = token.quality
        # 反熵催化：质量提升 ×5
        delta = entropy_reverse_power * self.PURIFY_COEFFICIENT * 5
        delta = min(delta, self.QUALITY_MAX - q_before)
        q_after = q_before + delta
        token.quality = q_after

        return {
            "action": "entropy_reverser_catalyze",
            "culture": "entropy_reverser",
            "culture_level": culture_level,
            "quality_before": round(q_before, 4),
            "quality_after": round(q_after, 4),
            "delta": round(delta, 4),
            "purity_before_pct": round(q_before / self.QUALITY_MAX * 100, 4),
            "purity_after_pct": round(q_after / self.QUALITY_MAX * 100, 4),
        }


# ============================================================
# 便捷函数
# ============================================================

def demo_purity_climb(cycles: int = 100, verbose: bool = True):
    """演示：100 次提纯看能否从 1% 爬到 10%"""
    factory = MultiverseResourceFactory()
    token = factory.produce_download_token()
    forge = PurityForge(factory)

    print("=" * 60)
    print("精纯度锻造炉 · 100次提纯测试")
    print("=" * 60)
    print(f"初始 quality      : {token.quality}")
    print(f"初始 精纯度        : {token.quality / 100 * 100:.4f}%")
    print(f"目标 精纯度        : 10%")
    print("-" * 60)

    result = forge.purify_batch(token, cycles=cycles, verbose=verbose)

    print("-" * 60)
    print(f"最终 quality      : {result['final_quality']}")
    print(f"最终 精纯度        : {result['final_purity_pct']:.4f}%")
    print(f"目标 10% 是否达到  : {'是' if result['target_reached'] else '否'}")
    print(f"总增量            : +{result['total_delta']:.4f}")
    print(f"总损耗            : {result['total_loss']:.2f}")
    print(f"真实电力累计       : {result['real_power_total']:.6f}")
    print(f"耗时              : {result['elapsed_sec']}s")
    print("=" * 60)
    return result


if __name__ == "__main__":
    demo_purity_climb(cycles=100, verbose=True)
