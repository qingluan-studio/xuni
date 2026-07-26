"""
XuniOverseer — 训练监管看护模块

职责:
1. 实时监控训练指标（场能量、四维不变量、脑状态）
2. 检测异常（崩溃、静默、得分骤降、同步塌陷）
3. 自动干预保护训练进程
4. 生成监管报告

异常类型:
- FIELD_COLLAPSE: 场能量过低，采样点发电不足
- FIELD_OVERFLOW: 场能量溢出，可能导致权重爆炸
- SILENCE: 输出音频全零或极低能量
- SCORE_PLUNGE: 四维不变量综合得分骤降
- SYNC_COLLAPSE: 脑网络同步性完全塌陷
- WEIGHT_EXPLOSION: 连接权重超出合理范围
- STRATEGY_STUCK: 探索器陷入局部最优

干预措施:
- none: 无需干预
- boost_field: 增强虚拟电场注入
- dampen_field: 衰减场能量防止溢出
- reset_brain: 重置网络相位和振幅（保留权重）
- reset_full: 完全重置网络
- adjust_coupling: 调整场-脑耦合强度
- switch_strategy: 强制执行策略切换
- early_stop: 建议提前停止训练
"""

import time
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np


class AnomalyType(Enum):
    NONE = auto()
    FIELD_COLLAPSE = auto()
    FIELD_OVERFLOW = auto()
    SILENCE = auto()
    SCORE_PLUNGE = auto()
    SYNC_COLLAPSE = auto()
    WEIGHT_EXPLOSION = auto()
    STRATEGY_STUCK = auto()


class InterventionType(Enum):
    NONE = auto()
    BOOST_FIELD = auto()
    DAMPEN_FIELD = auto()
    RESET_BRAIN = auto()
    RESET_FULL = auto()
    ADJUST_COUPLING = auto()
    SWITCH_STRATEGY = auto()
    EARLY_STOP = auto()


@dataclass
class WatchResult:
    anomaly: AnomalyType
    severity: float
    intervention: InterventionType
    detail: str
    metrics: dict
    timestamp: float


@dataclass
class OverseerConfig:
    field_energy_collapse_threshold: float = 0.01
    field_energy_overflow_threshold: float = 100.0
    silence_rms_threshold: float = 0.0005
    score_plunge_ratio: float = 0.4
    sync_collapse_threshold: float = 0.05
    weight_explosion_threshold: float = 5.0
    strategy_stuck_window: int = 8

    window_size: int = 10

    auto_intervention: bool = True
    intervention_cooldown: int = 3
    max_consecutive_crashes: int = 5
    field_boost_multiplier: float = 2.0
    coupling_adjustment: float = 0.15


@dataclass
class AnomalyRecord:
    anomaly_type: AnomalyType
    epoch: int
    severity: float
    detail: str
    timestamp: float


class OverseerReport:
    def __init__(self):
        self.anomalies: List[AnomalyRecord] = []
        self.interventions: List[Dict] = []
        self.smooth_scores: List[float] = []
        self.smooth_field_energies: List[float] = []
        self.smooth_sync: List[float] = []
        self.strategy_switch_count: int = 0
        self.reset_count: int = 0
        self.early_stop_triggered: bool = False


class XuniOverseer:
    def __init__(self, config: Optional[OverseerConfig] = None):
        self.config = config or OverseerConfig()
        self.report = OverseerReport()

        self._field_energy_history: List[float] = []
        self._score_history: List[float] = []
        self._sync_history: List[float] = []
        self._w_mean_history: List[float] = []
        self._strategy_history: List[str] = []
        self._audio_rms_history: List[float] = []

        self._last_intervention_epoch: int = -999
        self._consecutive_crashes: int = 0
        self._strategy_no_improvement: Dict[str, int] = {}
        self._watch_log: List[WatchResult] = []

    def watch(
        self,
        epoch: int,
        field_energy: float,
        scores,
        brain_summary: dict,
        audio: Optional[np.ndarray] = None,
        current_strategy: str = "",
    ) -> WatchResult:
        if hasattr(scores, 'to_dict'):
            score_dict = scores.to_dict()
        elif isinstance(scores, dict):
            score_dict = scores
        else:
            score_dict = {"overall": 0.0, "itc": 0.0, "scs": 0.0, "iec": 0.0, "pfft": 0.0}

        overall = float(score_dict.get("overall", 0.0))
        itc = float(score_dict.get("itc", 0.0))
        scs = float(score_dict.get("scs", 0.0))
        iec = float(score_dict.get("iec", 0.0))
        pfft = float(score_dict.get("pfft", 0.0))
        sync = float(brain_summary.get("synchronization", 0.0))
        w_mean = float(brain_summary.get("W_mean", 0.0))
        w_max = float(brain_summary.get("W_max", 0.0))

        self._field_energy_history.append(field_energy)
        self._score_history.append(overall)
        self._sync_history.append(sync)
        self._w_mean_history.append(w_mean)
        if current_strategy:
            self._strategy_history.append(current_strategy)

        if audio is not None and len(audio) > 0:
            rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
            self._audio_rms_history.append(rms)
        else:
            rms = None

        anomaly = self._detect_anomaly(field_energy, overall, itc, scs, iec, pfft,
                                       sync, w_mean, w_max, rms, current_strategy, epoch)
        intervention = self._decide_intervention(anomaly, epoch, overall, sync)

        result = WatchResult(
            anomaly=anomaly if anomaly != AnomalyType.NONE else AnomalyType.NONE,
            severity=self._severity(anomaly, overall, sync, w_mean),
            intervention=intervention,
            detail=self._describe(anomaly),
            metrics={
                "epoch": epoch,
                "field_energy": round(field_energy, 4),
                "overall": round(overall, 4),
                "itc": round(itc, 4),
                "scs": round(scs, 4),
                "iec": round(iec, 4),
                "pfft": round(pfft, 4),
                "sync": round(sync, 4),
                "W_mean": round(w_mean, 4),
                "W_max": round(w_max, 4),
                "audio_rms": round(rms, 6) if rms is not None else None,
            },
            timestamp=time.time(),
        )

        self._watch_log.append(result)
        self._record_anomaly(result, epoch)

        return result

    def _detect_anomaly(
        self,
        field_energy: float,
        overall: float,
        itc: float,
        scs: float,
        iec: float,
        pfft: float,
        sync: float,
        w_mean: float,
        w_max: float,
        rms: Optional[float],
        strategy: str,
        epoch: int,
    ) -> AnomalyType:
        cfg = self.config

        if not np.isfinite(field_energy) or field_energy <= cfg.field_energy_collapse_threshold:
            return AnomalyType.FIELD_COLLAPSE

        if not np.isfinite(field_energy) or field_energy > cfg.field_energy_overflow_threshold:
            return AnomalyType.FIELD_OVERFLOW

        if rms is not None and rms < cfg.silence_rms_threshold:
            return AnomalyType.SILENCE

        if not np.isfinite(w_max) or w_max > cfg.weight_explosion_threshold:
            return AnomalyType.WEIGHT_EXPLOSION

        if len(self._score_history) >= 3:
            recent_avg = np.mean(self._score_history[-3:])
            if overall < recent_avg * cfg.score_plunge_ratio and recent_avg > 0.1:
                return AnomalyType.SCORE_PLUNGE

        if len(self._sync_history) >= 2:
            if sync < cfg.sync_collapse_threshold:
                return AnomalyType.SYNC_COLLAPSE

        if strategy and len(self._strategy_history) >= cfg.strategy_stuck_window:
            recent = self._strategy_history[-cfg.strategy_stuck_window:]
            if all(s == strategy for s in recent):
                recent_scores = self._score_history[-cfg.strategy_stuck_window:]
                if max(recent_scores) == min(recent_scores) or max(recent_scores) < 0.3:
                    return AnomalyType.STRATEGY_STUCK

        return AnomalyType.NONE

    def _severity(self, anomaly: AnomalyType, overall: float, sync: float, w_mean: float) -> float:
        if anomaly == AnomalyType.NONE:
            return 0.0

        base = {
            AnomalyType.FIELD_COLLAPSE: 0.8,
            AnomalyType.FIELD_OVERFLOW: 0.6,
            AnomalyType.SILENCE: 0.9,
            AnomalyType.SCORE_PLUNGE: 0.7,
            AnomalyType.SYNC_COLLAPSE: 0.7,
            AnomalyType.WEIGHT_EXPLOSION: 0.9,
            AnomalyType.STRATEGY_STUCK: 0.5,
        }.get(anomaly, 0.5)

        if anomaly == AnomalyType.SCORE_PLUNGE:
            base *= 1.0 + (1.0 - min(overall, 1.0))
        if anomaly == AnomalyType.SYNC_COLLAPSE:
            base *= 1.0 + (1.0 - min(sync, 1.0))

        return min(base, 1.0)

    def _describe(self, anomaly: AnomalyType) -> str:
        descriptions = {
            AnomalyType.FIELD_COLLAPSE: "场能量过低，采样点发电不足，检查采样器参数",
            AnomalyType.FIELD_OVERFLOW: "场能量溢出，可能导致数值不稳定",
            AnomalyType.SILENCE: "输出音频静默，网络可能已死亡",
            AnomalyType.SCORE_PLUNGE: "音乐质量得分骤降，网络可能失去音乐性",
            AnomalyType.SYNC_COLLAPSE: "脑网络同步性完全塌陷，振荡器失协",
            AnomalyType.WEIGHT_EXPLOSION: "连接权重爆炸，需要重置网络",
            AnomalyType.STRATEGY_STUCK: "探索器陷入局部最优，建议强制切换策略",
            AnomalyType.NONE: "",
        }
        return descriptions.get(anomaly, "")

    def _decide_intervention(
        self, anomaly: AnomalyType, epoch: int, overall: float, sync: float
    ) -> InterventionType:
        if anomaly == AnomalyType.NONE:
            return InterventionType.NONE

        if not self.config.auto_intervention:
            return InterventionType.NONE

        if epoch - self._last_intervention_epoch < self.config.intervention_cooldown:
            return InterventionType.NONE

        if self._consecutive_crashes >= self.config.max_consecutive_crashes:
            return InterventionType.EARLY_STOP

        mapping = {
            AnomalyType.FIELD_COLLAPSE: InterventionType.BOOST_FIELD,
            AnomalyType.FIELD_OVERFLOW: InterventionType.DAMPEN_FIELD,
            AnomalyType.SILENCE: InterventionType.RESET_FULL,
            AnomalyType.SCORE_PLUNGE: InterventionType.RESET_BRAIN,
            AnomalyType.SYNC_COLLAPSE: InterventionType.ADJUST_COUPLING,
            AnomalyType.WEIGHT_EXPLOSION: InterventionType.RESET_FULL,
            AnomalyType.STRATEGY_STUCK: InterventionType.SWITCH_STRATEGY,
        }

        return mapping.get(anomaly, InterventionType.NONE)

    def _record_anomaly(self, result: WatchResult, epoch: int):
        if result.anomaly != AnomalyType.NONE:
            self.report.anomalies.append(AnomalyRecord(
                anomaly_type=result.anomaly,
                epoch=epoch,
                severity=result.severity,
                detail=result.detail,
                timestamp=result.timestamp,
            ))
            self._consecutive_crashes += 1

            if result.intervention != InterventionType.NONE:
                self._last_intervention_epoch = epoch
                self.report.interventions.append({
                    "epoch": epoch,
                    "intervention": result.intervention.name,
                    "severity": result.severity,
                    "field_energy": result.metrics.get("field_energy"),
                    "overall": result.metrics.get("overall"),
                })

            if result.intervention in (InterventionType.RESET_FULL, InterventionType.RESET_BRAIN):
                self.report.reset_count += 1
            if result.intervention == InterventionType.SWITCH_STRATEGY:
                self.report.strategy_switch_count += 1
            if result.intervention == InterventionType.EARLY_STOP:
                self.report.early_stop_triggered = True
        else:
            self._consecutive_crashes = 0

    def should_early_stop(self) -> bool:
        return (
            self.report.early_stop_triggered
            or self._consecutive_crashes >= self.config.max_consecutive_crashes
        )

    def get_intervention_params(self, intervention: InterventionType, current_params: dict) -> dict:
        params = {"intervention": intervention.name}

        if intervention == InterventionType.BOOST_FIELD:
            params["field_multiplier"] = self.config.field_boost_multiplier
        elif intervention == InterventionType.DAMPEN_FIELD:
            params["field_multiplier"] = 0.3
        elif intervention == InterventionType.RESET_BRAIN:
            params["preserve_weights"] = True
        elif intervention == InterventionType.RESET_FULL:
            params["preserve_weights"] = False
        elif intervention == InterventionType.ADJUST_COUPLING:
            params["coupling_delta"] = self.config.coupling_adjustment
        elif intervention == InterventionType.SWITCH_STRATEGY:
            params["force_new"] = True

        return params

    def execute_intervention(
        self,
        intervention: InterventionType,
        brain,
        field,
        explorer,
    ) -> Dict:
        result = {"action": intervention.name, "details": {}}

        if intervention == InterventionType.RESET_BRAIN:
            old_weights = brain.W.copy() if hasattr(brain, 'W') else None
            brain.reset()
            if old_weights is not None and hasattr(brain, 'W'):
                brain.W = old_weights
                if hasattr(brain, 'W_structural'):
                    brain.W_structural = old_weights.copy()
            result["details"] = {"weights_preserved": True}

        elif intervention == InterventionType.RESET_FULL:
            brain.reset()
            if hasattr(brain, 'W'):
                rng = np.random.default_rng()
                n = brain.n if hasattr(brain, 'n') else 256
                density = brain.connection_density if hasattr(brain, 'connection_density') else 0.3
                mask = (rng.random((n, n)) < density).astype(np.float64)
                brain.W = mask * rng.normal(0, 0.1, (n, n))
                np.fill_diagonal(brain.W, 0)
                if hasattr(brain, 'W_structural'):
                    brain.W_structural = brain.W.copy()
            result["details"] = {"full_reset": True}

        elif intervention == InterventionType.BOOST_FIELD:
            result["details"] = {"field_multiplier": self.config.field_boost_multiplier}

        elif intervention == InterventionType.DAMPEN_FIELD:
            result["details"] = {"field_multiplier": 0.3}

        elif intervention == InterventionType.ADJUST_COUPLING:
            if hasattr(brain, 'field_coupling'):
                old_coupling = brain.field_coupling
                brain.field_coupling += self.config.coupling_adjustment
                brain.field_coupling = min(brain.field_coupling, 2.0)
                result["details"] = {
                    "old_coupling": old_coupling,
                    "new_coupling": brain.field_coupling,
                }

        elif intervention == InterventionType.SWITCH_STRATEGY:
            result["details"] = {"force_strategy_switch": True}

        elif intervention == InterventionType.EARLY_STOP:
            result["details"] = {"early_stop": True, "reason": "max_consecutive_crashes"}

        self._consecutive_crashes = 0
        return result

    def get_safety_report(self) -> dict:
        smoothed_scores = []
        smoothed_fields = []
        smoothed_sync = []

        if self._score_history:
            w = min(self.config.window_size, len(self._score_history))
            kernel = np.ones(w) / w
            smoothed_scores = np.convolve(self._score_history, kernel, mode='valid').tolist()
            smoothed_fields = np.convolve(self._field_energy_history, kernel, mode='valid').tolist()
            smoothed_sync = np.convolve(self._sync_history, kernel, mode='valid').tolist()

        return {
            "total_anomalies": len(self.report.anomalies),
            "total_interventions": len(self.report.interventions),
            "resets": self.report.reset_count,
            "strategy_switches": self.report.strategy_switch_count,
            "early_stopped": self.report.early_stop_triggered,
            "consecutive_crashes": self._consecutive_crashes,
            "current_score_trend": smoothed_scores[-1] if smoothed_scores else None,
            "current_field_trend": smoothed_fields[-1] if smoothed_fields else None,
            "current_sync_trend": smoothed_sync[-1] if smoothed_sync else None,
            "anomaly_summary": self._summarize_anomalies(),
            "intervention_summary": self._summarize_interventions(),
        }

    def _summarize_anomalies(self) -> dict:
        counts = {}
        for a in self.report.anomalies:
            name = a.anomaly_type.name
            counts[name] = counts.get(name, 0) + 1
        return counts

    def _summarize_interventions(self) -> dict:
        counts = {}
        for i in self.report.interventions:
            name = i["intervention"]
            counts[name] = counts.get(name, 0) + 1
        return counts

    def print_watch_summary(self, result: WatchResult):
        if result.anomaly == AnomalyType.NONE:
            return

        icon = "[WARN]" if result.severity < 0.7 else "[CRITICAL]"
        lines = [
            f"  {icon} 监管告警 | Epoch {result.metrics['epoch']} | 严重度: {result.severity:.1%}",
            f"        类型: {result.anomaly.name} | {result.detail}",
            f"        干预: {result.intervention.name}",
        ]
        for line in lines:
            print(line)

    def print_final_report(self):
        report = self.get_safety_report()
        print(f"""
============================================================
  监管看护终验报告
============================================================
  异常事件: {report['total_anomalies']}  干预次数: {report['total_interventions']}
  网络重置: {report['resets']}          策略切换: {report['strategy_switches']}
  提前停止: {'是' if report['early_stopped'] else '否'}
  连续崩溃: {report['consecutive_crashes']}
""")
        if report['anomaly_summary']:
            print("  异常分布:")
            for name, count in report['anomaly_summary'].items():
                print(f"    {name}: {count}")
        if report['intervention_summary']:
            print("  干预分布:")
            for name, count in report['intervention_summary'].items():
                print(f"    {name}: {count}")
        if report['current_score_trend'] is not None:
            print(f"  得分趋势: {report['current_score_trend']:.4f}")
            print(f"  场能趋势: {report['current_field_trend']:.4f}")
            print(f"  同步趋势: {report['current_sync_trend']:.4f}")
        print("============================================================\n")
