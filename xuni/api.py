"""
XuniAPI —— RESTful API & WebSocket 实时流

提供 HTTP 接口用于：
- 生成采样点
- 计算虚拟电场
- 合成音乐
- 实时参数流（WebSocket）

手机可用：响应式 Web 控制面板内嵌在 / 路由。
"""

import numpy as np
import asyncio
from typing import Optional, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .sampler import XuniSampler, SamplingMode, SamplePoint
from .field import XuniField
from .converter import XuniConverter, MusicParams
from .music import XuniMusic, AudioBuffer
from .brain import XuniBrain
from .trainer import XuniTrainer, TrainingConfig
from .memory import XuniMemory
from .critic import XuniCritic, MusicInvariantScores
from .explorer import XuniExplorer, SamplingStrategy
from .overseer import XuniOverseer, OverseerConfig
from .layer import LayeredModelSystem, LayerType, LayerConfig, AI_NAME_POOL
from .model import ModelInput, TrainingState


# ------------------------------------------------------------------
# Pydantic 模型
# ------------------------------------------------------------------
class GenerateRequest(BaseModel):
    mode: str = "hyper_chaos"
    count: int = 10000
    seed: int = 42


class FieldRequest(BaseModel):
    mode: str = "hyper_chaos"
    sample_count: int = 100000
    grid_size: int = 32
    seed: int = 42


class MusicRequest(BaseModel):
    mode: str = "hyper_chaos"
    sample_count: int = 100000
    duration: float = 5.0
    seed: int = 42


class BrainGenerateRequest(BaseModel):
    mode: str = "hyper_chaos"
    sample_count: int = 100000
    duration: float = 5.0
    seed: int = 42
    n_neurons: int = 256
    field_coupling: float = 0.5


class BrainCultivateRequest(BaseModel):
    mode: str = "hyper_chaos"
    sample_count: int = 100000
    duration: float = 5.0
    seed: int = 42
    n_neurons: int = 256
    field_coupling: float = 0.5
    epochs: int = 1


class CriticRequest(BaseModel):
    mode: str = "hyper_chaos"
    sample_count: int = 100000
    duration: float = 3.0
    seed: int = 42


class ExploreRequest(BaseModel):
    n_trials: int = 5
    duration: float = 3.0


class FieldCultivateRequest(BaseModel):
    mode: str = "hyper_chaos"
    sample_count: int = 100000
    duration: float = 5.0
    seed: int = 42
    n_neurons: int = 256
    field_coupling: float = 0.6
    epochs: int = 3


class MassiveTrainRequest(BaseModel):
    mode: str = "quick"
    total_samples: int = 1_000_000
    batch_size: int = 200_000
    n_neurons: int = 256
    grid_size: int = 24
    training_epochs: int = 30
    music_duration: float = 5.0
    seed: int = 42
    auto_intervention: bool = True


class MemoryQueryRequest(BaseModel):
    memory_type: str = "resonance"
    top_k: int = 10


class StatusResponse(BaseModel):
    status: str
    version: str
    capabilities: List[str]


# ------------------------------------------------------------------
# FastAPI 应用
# ------------------------------------------------------------------
app = FastAPI(
    title="Xuni API",
    description="虚拟电场与音乐生成系统 API",
    version="0.1.0",
)

# 状态存储（简单内存存储，生产环境可换 Redis）
app_state = {
    "sessions": {},
    "massive_training": {
        "running": False,
        "progress": 0,
        "total_epochs": 0,
        "best_score": 0.0,
        "anomalies": 0,
    },
}


def _parse_mode(mode_str: str) -> SamplingMode:
    mapping = {
        "hyper_chaos": SamplingMode.HYPER_CHAOS,
        "lorenz_96": SamplingMode.LORENZ_96,
        "mandelbulb": SamplingMode.MANDELBULB,
        "noise_field": SamplingMode.NOISE_FIELD,
        "hybrid": SamplingMode.HYBRID,
    }
    return mapping.get(mode_str.lower(), SamplingMode.HYPER_CHAOS)


# ------------------------------------------------------------------
# 路由
# ------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    """返回手机控制面板"""
    return HTMLResponse(content=_WEB_UI_HTML)


@app.get("/api/status", response_model=StatusResponse)
async def status():
    return StatusResponse(
        status="running",
        version="0.1.0",
        capabilities=[
            "sample_generation",
            "field_computation",
            "music_synthesis",
            "websocket_streaming",
            "brain_resonance",
            "brain_cultivation",
            "brain_field_cultivation",
            "massive_training",
            "critic_evaluation",
            "explorer_optimization",
            "overseer_monitoring",
            "memory_recall",
        ],
    )


@app.post("/api/sample/generate")
async def generate_samples(req: GenerateRequest):
    """生成采样点（返回摘要统计）"""
    mode = _parse_mode(req.mode)
    sampler = XuniSampler(mode=mode, seed=req.seed)

    batch = sampler.generate_batch(req.count)

    return {
        "mode": req.mode,
        "count": req.count,
        "seed": req.seed,
        "summary": {
            "x_range": [float(batch[:,0].min()), float(batch[:,0].max())],
            "y_range": [float(batch[:,1].min()), float(batch[:,1].max())],
            "z_range": [float(batch[:,2].min()), float(batch[:,2].max())],
            "mean_charge": float(batch[:,4].mean()),
            "mean_entropy": float(batch[:,5].mean()),
        },
        "preview": batch[:10].tolist(),  # 前 10 个点预览
    }


@app.post("/api/field/compute")
async def compute_field(req: FieldRequest):
    """计算虚拟电场"""
    mode = _parse_mode(req.mode)
    sampler = XuniSampler(mode=mode, seed=req.seed)
    field = XuniField(grid_size=(req.grid_size, req.grid_size, req.grid_size))

    # 流式吸收采样点
    stream = sampler.generate_stream(req.sample_count)
    field.ingest_stream(stream, req.sample_count)
    field.compute_field()

    summary = field.field_summary()
    cells = field.get_cells(threshold=0.1)

    return {
        "mode": req.mode,
        "sample_count": req.sample_count,
        "grid_size": req.grid_size,
        "field_summary": summary,
        "high_energy_cells": len(cells),
        "dominant_vector": field.get_dominant_vector(),
    }


@app.post("/api/music/generate")
async def generate_music(req: MusicRequest):
    """生成音乐（返回 WAV 音频流）"""
    mode = _parse_mode(req.mode)
    sampler = XuniSampler(mode=mode, seed=req.seed)
    field = XuniField(grid_size=(16, 16, 16))
    converter = XuniConverter()
    music = XuniMusic(sample_rate=22050)

    # 1. 生成采样点并构建场
    batch = sampler.generate_batch(req.sample_count)
    field.ingest_batch(batch)
    field.compute_field()

    # 2. 转换为音乐参数
    summary = field.field_summary()
    summary["dominant_ex"] = field.get_dominant_vector()[0]
    summary["dominant_ey"] = field.get_dominant_vector()[1]
    summary["dominant_ez"] = field.get_dominant_vector()[2]
    energy_dist = field.get_energy_distribution()
    params = converter.convert(summary, energy_dist)

    # 3. 合成音乐
    audio = music.synthesize(params, duration=req.duration)
    wav_bytes = music.to_wav_bytes(audio)

    return StreamingResponse(
        iter([wav_bytes]),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=xuni_music.wav"},
    )


@app.get("/api/music/generate_seq")
async def generate_music_sequence(
    mode: str = "hyper_chaos",
    segments: int = 4,
    segment_duration: float = 2.0,
    samples_per_segment: int = 50000,
    seed: int = 42,
):
    """生成音乐序列（多段拼接）"""
    mode_enum = _parse_mode(mode)
    music = XuniMusic(sample_rate=22050)
    converter = XuniConverter()
    params_list = []

    for i in range(segments):
        sampler = XuniSampler(mode=mode_enum, seed=seed + i)
        field = XuniField(grid_size=(16, 16, 16))
        batch = sampler.generate_batch(samples_per_segment)
        field.ingest_batch(batch)
        field.compute_field()

        summary = field.field_summary()
        summary["dominant_ex"] = field.get_dominant_vector()[0]
        summary["dominant_ey"] = field.get_dominant_vector()[1]
        summary["dominant_ez"] = field.get_dominant_vector()[2]
        energy_dist = field.get_energy_distribution()
        params_list.append(converter.convert(summary, energy_dist))

    audio = music.synthesize_sequence(params_list, segment_duration=segment_duration)
    wav_bytes = music.to_wav_bytes(audio)

    return StreamingResponse(
        iter([wav_bytes]),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=xuni_sequence.wav"},
    )


@app.post("/api/brain/generate")
async def generate_brain_music(req: BrainGenerateRequest):
    """场能量驱动 Brain 生成音乐（共振神经网络）"""
    mode = _parse_mode(req.mode)
    sampler = XuniSampler(mode=mode, seed=req.seed)
    field = XuniField(grid_size=(16, 16, 16))

    # 1. 生成采样点并构建场
    batch = sampler.generate_batch(req.sample_count)
    field.ingest_batch(batch)
    field.compute_field()

    # 2. 提取场能量
    summary = field.field_summary()
    energy_dist = field.get_energy_distribution()
    total_energy = summary.get("total_energy", 1.0)
    field_energy = np.log1p(total_energy)

    # 3. 创建 Brain 并用场能量刺激
    brain = XuniBrain(
        n_neurons=req.n_neurons,
        sample_rate=22050,
        field_coupling=req.field_coupling,
        seed=req.seed,
    )
    audio = brain.stimulate(duration=req.duration, field_energy=field_energy)

    # 4. 包装为 WAV
    music = XuniMusic(sample_rate=22050)
    buf = AudioBuffer(sample_rate=22050, data=audio, duration=req.duration)
    wav_bytes = music.to_wav_bytes(buf)

    return StreamingResponse(
        iter([wav_bytes]),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=xuni_brain.wav"},
    )


@app.post("/api/brain/cultivate")
async def cultivate_brain(req: BrainCultivateRequest):
    """培养 Brain：用场驱动的音乐作为目标进行共振学习"""
    mode = _parse_mode(req.mode)
    sampler = XuniSampler(mode=mode, seed=req.seed)
    field = XuniField(grid_size=(16, 16, 16))
    converter = XuniConverter()
    music = XuniMusic(sample_rate=22050)

    # 1. 生成目标音乐（场 → 参数 → 音频）
    batch = sampler.generate_batch(req.sample_count)
    field.ingest_batch(batch)
    field.compute_field()
    summary = field.field_summary()
    summary["dominant_ex"] = field.get_dominant_vector()[0]
    summary["dominant_ey"] = field.get_dominant_vector()[1]
    summary["dominant_ez"] = field.get_dominant_vector()[2]
    energy_dist = field.get_energy_distribution()
    params = converter.convert(summary, energy_dist)
    target_audio = music.synthesize(params, duration=req.duration).to_mono()

    # 2. 创建 Brain 并培养
    total_energy = summary.get("total_energy", 1.0)
    field_energy = np.log1p(total_energy)
    brain = XuniBrain(
        n_neurons=req.n_neurons,
        sample_rate=22050,
        field_coupling=req.field_coupling,
        seed=req.seed,
    )
    trainer = XuniTrainer(brain, config=TrainingConfig())
    output = trainer.cultivate(
        target_audio=target_audio,
        duration=req.duration,
        field_energy=field_energy,
        epochs=req.epochs,
    )

    # 3. 返回培养后的自发输出
    buf = AudioBuffer(sample_rate=22050, data=output, duration=req.duration)
    wav_bytes = music.to_wav_bytes(buf)

    return StreamingResponse(
        iter([wav_bytes]),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=xuni_cultivated.wav"},
    )


@app.post("/api/brain/cultivate_field")
async def cultivate_brain_from_field(req: FieldCultivateRequest):
    """纯场能量驱动 Brain 自培养（无需目标音频，场能直接注入 Hebbian 学习）"""
    mode = _parse_mode(req.mode)
    sampler = XuniSampler(mode=mode, seed=req.seed)
    field = XuniField(grid_size=(16, 16, 16))

    batch = sampler.generate_batch(req.sample_count)
    field.ingest_batch(batch)
    field.compute_field()

    total_energy = field.field_summary().get("total_energy", 1.0)
    field_energy = float(np.log1p(total_energy) * 0.5)

    brain = XuniBrain(
        n_neurons=req.n_neurons,
        sample_rate=22050,
        field_coupling=req.field_coupling,
        seed=req.seed,
    )
    trainer = XuniTrainer(brain, config=TrainingConfig())

    output = trainer.cultivate_from_field(
        field_energy=field_energy,
        duration=req.duration,
        epochs=req.epochs,
    )

    music = XuniMusic(sample_rate=22050)
    buf = AudioBuffer(sample_rate=22050, data=output, duration=req.duration)
    wav_bytes = music.to_wav_bytes(buf)

    return StreamingResponse(
        iter([wav_bytes]),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=xuni_cultivated_field.wav"},
    )


@app.get("/api/brain/status")
async def brain_status(
    n_neurons: int = 256,
    field_coupling: float = 0.5,
    seed: int = 42,
):
    """获取 Brain 状态摘要"""
    brain = XuniBrain(
        n_neurons=n_neurons,
        sample_rate=22050,
        field_coupling=field_coupling,
        seed=seed,
    )
    return {
        "status": "ready",
        "n_neurons": brain.n,
        "mean_freq": float(np.mean(brain.freq)),
        "freq_range": [float(brain.freq.min()), float(brain.freq.max())],
        "mean_amp": float(np.mean(brain.amp)),
        "synchronization": float(np.abs(np.mean(np.exp(1j * brain.phi)))),
        "field_coupling": brain.field_coupling,
    }


@app.post("/api/critic/evaluate")
async def evaluate_music(req: CriticRequest):
    """评估生成的音乐质量（四维认知不变量）"""
    mode = _parse_mode(req.mode)
    sampler = XuniSampler(mode=mode, seed=req.seed)
    field = XuniField(grid_size=(16, 16, 16))
    converter = XuniConverter()
    music = XuniMusic(sample_rate=22050)

    batch = sampler.generate_batch(req.sample_count)
    field.ingest_batch(batch)
    field.compute_field()
    summary = field.field_summary()
    summary["dominant_ex"] = field.get_dominant_vector()[0]
    summary["dominant_ey"] = field.get_dominant_vector()[1]
    summary["dominant_ez"] = field.get_dominant_vector()[2]
    params = converter.convert(summary, field.get_energy_distribution())
    audio = music.synthesize(params, duration=req.duration).to_mono()

    critic = XuniCritic(sample_rate=22050)
    scores = critic.evaluate(audio)
    suggestions = critic.suggest_optimization(scores)

    return {
        "scores": scores.to_dict(),
        "suggestions": suggestions,
        "params": {
            "base_frequency": params.base_frequency,
            "harmonics": params.harmonics,
            "tempo": params.tempo,
        },
    }


@app.post("/api/explorer/run")
async def run_exploration(req: ExploreRequest):
    """运行探索-利用循环，自动寻找最佳参数组合"""
    explorer = XuniExplorer(epsilon=0.4)
    for mode in ["hyper_chaos", "lorenz_96", "mandelbulb", "noise_field", "hybrid"]:
        explorer.register_sampling_mode(SamplingStrategy(mode))

    music = XuniMusic(sample_rate=22050)
    converter = XuniConverter()
    critic = XuniCritic(sample_rate=22050)

    results = []
    for trial in range(req.n_trials):
        name, params = explorer.select_strategy(category="sample")
        mode_str = params["mode"]
        mode = _parse_mode(mode_str)
        sampler = XuniSampler(mode=mode, seed=trial)
        field = XuniField(grid_size=(16, 16, 16))
        batch = sampler.generate_batch(50000)
        field.ingest_batch(batch)
        field.compute_field()
        summary = field.field_summary()
        summary["dominant_ex"] = field.get_dominant_vector()[0]
        summary["dominant_ey"] = field.get_dominant_vector()[1]
        summary["dominant_ez"] = field.get_dominant_vector()[2]
        mp = converter.convert(summary, field.get_energy_distribution())
        audio = music.synthesize(mp, duration=req.duration).to_mono()
        scores = critic.evaluate(audio)
        explorer.feedback(name, scores.overall)
        results.append({
            "trial": trial,
            "mode": mode_str,
            "overall": scores.overall,
        })

    report = explorer.get_report()
    return {
        "trials": results,
        "report": report,
    }


@app.post("/api/massive/train")
async def start_massive_training(req: MassiveTrainRequest):
    """启动海量训练任务，采样点发电驱动大规模自培养"""
    if app_state["massive_training"]["running"]:
        return JSONResponse(
            status_code=409,
            content={"error": "海量训练已在运行中", "status": app_state["massive_training"]},
        )

    app_state["massive_training"] = {
        "running": True,
        "progress": 0,
        "total_epochs": req.training_epochs,
        "best_score": 0.0,
        "anomalies": 0,
    }

    import threading

    def _run_training():
        try:
            rng = np.random.default_rng(req.seed)
            field = XuniField(
                grid_size=(req.grid_size, req.grid_size, req.grid_size),
                bounds=(-80.0, 80.0),
                smooth_sigma=1.8,
            )
            brain = XuniBrain(
                n_neurons=req.n_neurons,
                sample_rate=22050,
                connection_density=0.35,
                field_coupling=0.6,
                seed=req.seed,
            )
            trainer = XuniTrainer(brain, config=TrainingConfig(
                hebbian_rate=0.002,
                weight_decay=0.00005,
                field_boost=3.0,
            ))
            critic = XuniCritic(sample_rate=22050)
            explorer = XuniExplorer(epsilon=0.4, use_thompson=True, use_novelty=True)
            for s in SamplingStrategy:
                explorer.register_sampling_mode(s)

            overseer = XuniOverseer(OverseerConfig(
                auto_intervention=req.auto_intervention,
                intervention_cooldown=3,
                max_consecutive_crashes=8,
            ))
            memory = XuniMemory(brain)

            batch_count = max(1, req.total_samples // req.batch_size)
            best_score = 0.0
            best_audio = None

            for epoch in range(req.training_epochs):
                if not app_state["massive_training"]["running"]:
                    break

                strategy_name, _ = explorer.select_strategy("sample")
                mode_enum = SamplingMode[strategy_name.replace("sample_", "").upper()]
                field.reset()

                sampler = XuniSampler(mode=mode_enum, seed=rng.integers(0, 100000))
                total_energy = 0.0
                for _ in range(batch_count):
                    batch = sampler.generate_batch(req.batch_size)
                    field.ingest_batch(batch)
                field.compute_field()
                summary = field.field_summary()
                total_energy = summary.get("total_energy", 1.0)
                field_energy = float(np.log1p(total_energy) * 0.5)

                audio = trainer.cultivate_from_field(
                    field_energy=field_energy,
                    duration=req.music_duration,
                    epochs=1,
                )

                scores = critic.evaluate(audio)
                explorer.feedback(strategy_name, scores.overall)
                if scores.overall > best_score:
                    best_score = scores.overall
                    best_audio = audio.copy()

                watch = overseer.watch(
                    epoch=epoch,
                    field_energy=field_energy,
                    scores=scores,
                    brain_summary=brain.brain_summary(),
                    audio=audio,
                    current_strategy=strategy_name,
                )
                if watch.anomaly.value != 0:
                    app_state["massive_training"]["anomalies"] += 1
                    if watch.intervention.value != 1:
                        from xuni.overseer import InterventionType
                        overseer.execute_intervention(
                            watch.intervention, brain, field, explorer
                        )

                if overseer.should_early_stop():
                    break

                if epoch % 5 == 0:
                    memory.capture(f"epoch_{epoch}", audio)

                app_state["massive_training"].update({
                    "progress": epoch + 1,
                    "best_score": round(best_score, 4),
                    "safety_report": overseer.get_safety_report(),
                })

            app_state["massive_training"]["running"] = False
            app_state["massive_training"]["best_audio"] = best_audio

        except Exception as e:
            app_state["massive_training"]["running"] = False
            app_state["massive_training"]["error"] = str(e)

    threading.Thread(target=_run_training, daemon=True).start()

    return {
        "message": "海量训练已启动",
        "config": {
            "mode": req.mode,
            "total_samples": req.total_samples,
            "n_neurons": req.n_neurons,
            "grid_size": req.grid_size,
            "training_epochs": req.training_epochs,
        },
    }


@app.get("/api/massive/status")
async def get_massive_status():
    """获取海量训练当前状态"""
    state = app_state["massive_training"]
    return {
        "running": state["running"],
        "progress": state.get("progress", 0),
        "total_epochs": state.get("total_epochs", 0),
        "best_score": state.get("best_score", 0.0),
        "anomalies": state.get("anomalies", 0),
        "safety_report": state.get("safety_report"),
        "error": state.get("error"),
    }


@app.post("/api/massive/stop")
async def stop_massive_training():
    """停止海量训练"""
    was_running = app_state["massive_training"]["running"]
    app_state["massive_training"]["running"] = False
    return {
        "message": "训练已停止" if was_running else "训练未在运行",
        "final_status": app_state["massive_training"],
    }


@app.get("/api/massive/audio")
async def get_massive_audio():
    """获取海量训练的最佳音频输出"""
    best_audio = app_state["massive_training"].get("best_audio")
    if best_audio is None:
        return JSONResponse(status_code=404, content={"error": "无可用音频"})

    music = XuniMusic(sample_rate=22050)
    buf = AudioBuffer(sample_rate=22050, data=best_audio, duration=5.0)
    wav_bytes = music.to_wav_bytes(buf)

    return StreamingResponse(
        iter([wav_bytes]),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=xuni_massive_best.wav"},
    )


@app.get("/api/memory/list")
async def list_memory(req: MemoryQueryRequest = MemoryQueryRequest()):
    """列出记忆银行中的共振模式"""
    brain = XuniBrain(n_neurons=256, seed=0)
    memory = XuniMemory(brain)

    entries = []
    for m in memory.memories[:req.top_k]:
        entries.append({
            "name": m.name,
            "tags": m.tags,
            "evocations": m.evocations,
            "created_at": m.created_at,
            "duration": m.duration,
            "summary": {
                "mean_freq": round(float(np.mean(m.freq)), 2),
                "n_neurons": len(m.phi),
                "weight_mean": round(float(np.mean(np.abs(m.W))), 4),
            },
        })

    return {
        "count": len(entries),
        "total_memories": len(memory.memories),
        "entries": entries,
    }


@app.get("/api/overseer/health")
async def overseer_health():
    """获取 Brain 和监管的综合健康检查"""
    brain = XuniBrain(n_neurons=256, field_coupling=0.5, seed=42)
    summary = brain.brain_summary()
    return {
        "brain": {
            "n_neurons": summary["n_neurons"],
            "synchronization": round(summary["synchronization"], 4),
            "mean_freq": round(summary["mean_freq"], 2),
            "W_mean": round(summary["W_mean"], 4),
            "W_max": round(summary["W_max"], 4),
            "steps": summary["steps"],
        },
        "massive_training": {
            "running": app_state["massive_training"]["running"],
            "progress": f"{app_state['massive_training'].get('progress', 0)}/{app_state['massive_training'].get('total_epochs', 0)}",
        },
    }


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket 实时流。

    客户端发送 JSON 命令，服务器推送实时生成的采样点/场数据/音乐参数。
    """
    await websocket.accept()
    try:
        while True:
            msg = await websocket.receive_json()
            cmd = msg.get("cmd", "sample")
            mode = _parse_mode(msg.get("mode", "hyper_chaos"))
            seed = msg.get("seed", 42)
            count = msg.get("count", 1000)

            if cmd == "sample":
                sampler = XuniSampler(mode=mode, seed=seed)
                batch = sampler.generate_batch(count)
                await websocket.send_json({
                    "type": "sample",
                    "count": count,
                    "preview": batch[:20].tolist(),
                })

            elif cmd == "field":
                sampler = XuniSampler(mode=mode, seed=seed)
                field = XuniField(grid_size=(16, 16, 16))
                batch = sampler.generate_batch(count)
                field.ingest_batch(batch)
                field.compute_field()
                await websocket.send_json({
                    "type": "field",
                    "summary": field.field_summary(),
                    "dominant_vector": field.get_dominant_vector(),
                })

            elif cmd == "params":
                sampler = XuniSampler(mode=mode, seed=seed)
                field = XuniField(grid_size=(16, 16, 16))
                converter = XuniConverter()
                batch = sampler.generate_batch(count)
                field.ingest_batch(batch)
                field.compute_field()
                summary = field.field_summary()
                summary["dominant_ex"] = field.get_dominant_vector()[0]
                summary["dominant_ey"] = field.get_dominant_vector()[1]
                summary["dominant_ez"] = field.get_dominant_vector()[2]
                params = converter.convert(summary, field.get_energy_distribution())
                await websocket.send_json({
                    "type": "params",
                    "params": {
                        "base_frequency": params.base_frequency,
                        "amplitude": params.amplitude,
                        "harmonics": params.harmonics,
                        "tempo": params.tempo,
                        "filter_cutoff": params.filter_cutoff,
                        "noise_ratio": params.noise_ratio,
                    },
                })

            else:
                await websocket.send_json({"type": "error", "message": "Unknown cmd"})

    except WebSocketDisconnect:
        pass


# ------------------------------------------------------------------
# 嵌入式 Web 控制面板（手机响应式）
# ------------------------------------------------------------------
_WEB_UI_HTML = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
<title>Xuni 虚拟电场控制台</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #0a0a0f; color: #e0e0ff; font-family: -apple-system, BlinkMacSystemFont, sans-serif;
    min-height: 100vh; padding: 16px;
  }
  h1 { font-size: 1.4rem; text-align: center; margin-bottom: 8px; color: #00f0ff; }
  .subtitle { text-align: center; font-size: 0.8rem; color: #8888aa; margin-bottom: 20px; }
  .card {
    background: #12121f; border: 1px solid #1f1f3a; border-radius: 12px; padding: 16px; margin-bottom: 12px;
  }
  .card h2 { font-size: 1rem; margin-bottom: 12px; color: #aaccff; }
  label { display: block; font-size: 0.85rem; margin-bottom: 4px; color: #8899bb; }
  select, input[type="number"] {
    width: 100%; padding: 10px; border-radius: 8px; border: 1px solid #2a2a4a;
    background: #0d0d1a; color: #ccddee; font-size: 1rem; margin-bottom: 10px;
  }
  button {
    width: 100%; padding: 14px; border: none; border-radius: 10px;
    background: linear-gradient(135deg, #0066ff, #00ccff); color: white;
    font-size: 1rem; font-weight: 600; cursor: pointer; margin-top: 4px;
  }
  button:active { opacity: 0.8; }
  .btn-secondary { background: linear-gradient(135deg, #6600ff, #cc00ff); margin-top: 8px; }
  .output {
    background: #080810; border-radius: 8px; padding: 12px; font-family: monospace;
    font-size: 0.75rem; color: #55ff88; max-height: 200px; overflow-y: auto; margin-top: 10px;
    white-space: pre-wrap; word-break: break-all;
  }
  .stats { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 10px; }
  .stat { background: #0d0d1a; padding: 8px; border-radius: 6px; text-align: center; }
  .stat-value { font-size: 1.1rem; color: #00f0ff; font-weight: bold; }
  .stat-label { font-size: 0.7rem; color: #667799; }
  .loading { text-align: center; color: #00ccff; padding: 10px; }
  audio { width: 100%; margin-top: 10px; }
</style>
</head>
<body>
<h1>⚡ Xuni 虚拟电场</h1>
<p class="subtitle">超混沌采样 → 虚拟电荷 → 电场能量 → 原创音乐</p>

<div class="card">
  <h2>1. 采样配置</h2>
  <label>采样模式</label>
  <select id="mode">
    <option value="hyper_chaos">超混沌 Chen 系统</option>
    <option value="lorenz_96">Lorenz-96 高维环</option>
    <option value="mandelbulb">Mandelbulb 分形</option>
    <option value="noise_field">4D 噪声场</option>
    <option value="hybrid">混合模式</option>
  </select>
  <label>采样点数</label>
  <input type="number" id="count" value="100000" min="1000" max="10000000" step="1000">
  <label>随机种子</label>
  <input type="number" id="seed" value="42" min="0">
</div>

<div class="card">
  <h2>2. 操作</h2>
  <button onclick="doSample()">生成采样点</button>
  <button class="btn-secondary" onclick="doField()">计算虚拟电场</button>
  <button class="btn-secondary" onclick="doMusic()">合成音乐 (WAV)</button>
  <button class="btn-secondary" onclick="doSequence()">合成音乐序列</button>
  <button class="btn-secondary" onclick="doBrain()">Brain 共振生成</button>
  <button class="btn-secondary" onclick="doBrainCultivate()">Brain 培养</button>
  <button class="btn-secondary" onclick="doBrainCultivateField()">Brain 场培养</button>
  <button class="btn-secondary" onclick="doCritic()">评估音乐</button>
  <button class="btn-secondary" onclick="doExplore()">探索模式</button>
  <button class="btn-secondary" onclick="doMassiveTrain()">海量训练</button>
  <button class="btn-secondary" onclick="doMassiveStatus()">训练状态</button>
  <button class="btn-secondary" onclick="doOverseerHealth()">健康检查</button>
</div>

<div class="card">
  <h2>3. 结果</h2>
  <div id="result">
    <div class="stats">
      <div class="stat"><div class="stat-value" id="st-samples">-</div><div class="stat-label">采样点</div></div>
      <div class="stat"><div class="stat-value" id="st-energy">-</div><div class="stat-label">场能量</div></div>
      <div class="stat"><div class="stat-value" id="st-freq">-</div><div class="stat-label">频率(Hz)</div></div>
      <div class="stat"><div class="stat-value" id="st-tempo">-</div><div class="stat-label">BPM</div></div>
    </div>
    <div class="output" id="output">等待操作...</div>
    <div id="audio-container"></div>
  </div>
</div>

<script>
const API_BASE = '';

function getCfg() {
  return {
    mode: document.getElementById('mode').value,
    count: parseInt(document.getElementById('count').value),
    seed: parseInt(document.getElementById('seed').value),
  };
}

function setLoading(msg) {
  document.getElementById('output').textContent = msg;
}

async function doSample() {
  const cfg = getCfg();
  setLoading('正在生成采样点...');
  try {
    const res = await fetch(API_BASE + '/api/sample/generate', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({mode: cfg.mode, count: cfg.count, seed: cfg.seed})
    });
    const data = await res.json();
    document.getElementById('st-samples').textContent = data.count.toLocaleString();
    document.getElementById('output').textContent = JSON.stringify(data.summary, null, 2);
  } catch(e) { setLoading('错误: ' + e.message); }
}

async function doField() {
  const cfg = getCfg();
  setLoading('正在计算虚拟电场...');
  try {
    const res = await fetch(API_BASE + '/api/field/compute', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({mode: cfg.mode, sample_count: cfg.count, grid_size: 32, seed: cfg.seed})
    });
    const data = await res.json();
    document.getElementById('st-samples').textContent = data.sample_count.toLocaleString();
    document.getElementById('st-energy').textContent = data.field_summary.total_energy.toExponential(2);
    document.getElementById('output').textContent = JSON.stringify(data.field_summary, null, 2);
  } catch(e) { setLoading('错误: ' + e.message); }
}

async function doMusic() {
  const cfg = getCfg();
  setLoading('正在合成音乐...');
  try {
    const res = await fetch(API_BASE + '/api/music/generate', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({mode: cfg.mode, sample_count: cfg.count, duration: 5.0, seed: cfg.seed})
    });
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    document.getElementById('audio-container').innerHTML = `<audio controls src="${url}"></audio>`;
    setLoading('音乐已生成，点击播放 ↑');
  } catch(e) { setLoading('错误: ' + e.message); }
}

async function doSequence() {
  const cfg = getCfg();
  setLoading('正在合成音乐序列...');
  try {
    const res = await fetch(API_BASE + '/api/music/generate_seq?mode=' + cfg.mode
      + '&segments=4&segment_duration=2.0&samples_per_segment=' + Math.floor(cfg.count/4)
      + '&seed=' + cfg.seed);
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    document.getElementById('audio-container').innerHTML = `<audio controls src="${url}"></audio>`;
    setLoading('序列已生成，点击播放 ↑');
  } catch(e) { setLoading('错误: ' + e.message); }
}

async function doBrain() {
  const cfg = getCfg();
  setLoading('Brain 共振生成中...');
  try {
    const res = await fetch(API_BASE + '/api/brain/generate', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({mode: cfg.mode, sample_count: cfg.count, duration: 5.0, seed: cfg.seed})
    });
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    document.getElementById('audio-container').innerHTML = `<audio controls src="${url}"></audio>`;
    setLoading('Brain 音乐已生成，点击播放 ↑');
  } catch(e) { setLoading('错误: ' + e.message); }
}

async function doBrainCultivate() {
  const cfg = getCfg();
  setLoading('Brain 培养中（可能需要一点时间）...');
  try {
    const res = await fetch(API_BASE + '/api/brain/cultivate', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({mode: cfg.mode, sample_count: cfg.count, duration: 5.0, seed: cfg.seed, epochs: 1})
    });
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    document.getElementById('audio-container').innerHTML = `<audio controls src="${url}"></audio>`;
    setLoading('培养完成，点击播放');
  } catch(e) { setLoading('错误: ' + e.message); }
}

async function doBrainCultivateField() {
  const cfg = getCfg();
  setLoading('Brain 场培养中...');
  try {
    const res = await fetch(API_BASE + '/api/brain/cultivate_field', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({mode: cfg.mode, sample_count: cfg.count, duration: 5.0, seed: cfg.seed, epochs: 3})
    });
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    document.getElementById('audio-container').innerHTML = `<audio controls src="${url}"></audio>`;
    setLoading('场培养完成，点击播放');
  } catch(e) { setLoading('错误: ' + e.message); }
}

async function doMassiveTrain() {
  const cfg = getCfg();
  setLoading('海量训练启动中...');
  try {
    const res = await fetch(API_BASE + '/api/massive/train', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({
        total_samples: cfg.count,
        training_epochs: 30,
        seed: cfg.seed,
        n_neurons: 256,
        grid_size: 24,
      })
    });
    const data = await res.json();
    setLoading('海量训练: ' + JSON.stringify(data, null, 2));
  } catch(e) { setLoading('错误: ' + e.message); }
}

async function doMassiveStatus() {
  setLoading('获取训练状态...');
  try {
    const res = await fetch(API_BASE + '/api/massive/status');
    const data = await res.json();
    setLoading('训练状态: ' + JSON.stringify(data, null, 2));
  } catch(e) { setLoading('错误: ' + e.message); }
}

async function doOverseerHealth() {
  setLoading('健康检查中...');
  try {
    const res = await fetch(API_BASE + '/api/overseer/health');
    const data = await res.json();
    document.getElementById('output').textContent = '健康报告: ' + JSON.stringify(data, null, 2);
    setLoading('');
  } catch(e) { setLoading('错误: ' + e.message); }
}

async function doCritic() {
  const cfg = getCfg();
  setLoading('正在评估音乐质量...');
  try {
    const res = await fetch(API_BASE + '/api/critic/evaluate', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({mode: cfg.mode, sample_count: cfg.count, duration: 3.0, seed: cfg.seed})
    });
    const data = await res.json();
    document.getElementById('st-freq').textContent = data.params.base_frequency.toFixed(1);
    document.getElementById('st-tempo').textContent = data.params.tempo.toFixed(0);
    document.getElementById('output').textContent = '评分: ' + JSON.stringify(data.scores, null, 2)
      + '\n\n建议: ' + JSON.stringify(data.suggestions, null, 2);
  } catch(e) { setLoading('错误: ' + e.message); }
}

async function doExplore() {
  const cfg = getCfg();
  setLoading('探索模式中...');
  try {
    const res = await fetch(API_BASE + '/api/explorer/run', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({n_trials: 5, duration: 3.0})
    });
    const data = await res.json();
    document.getElementById('output').textContent = '探索报告: ' + JSON.stringify(data.report, null, 2)
      + '\n\n各次评分: ' + JSON.stringify(data.trials, null, 2);
  } catch(e) { setLoading('错误: ' + e.message); }
}
</script>
</body>
</html>
"""


# ------------------------------------------------------------------
# 分层模型系统 API
# ------------------------------------------------------------------

# 全局分层系统实例（支持持久化到文件）
_layer_system: Optional[LayeredModelSystem] = None
_layer_state_file = "xuni_layers.json"


def _get_layer_system() -> LayeredModelSystem:
    global _layer_system
    if _layer_system is None:
        _layer_system = LayeredModelSystem.load(_layer_state_file)
        if _layer_system is None:
            _layer_system = LayeredModelSystem()
            _layer_system.setup_default_layers()
    return _layer_system


# 分层系统监控仪表盘
@app.get("/layers", response_class=HTMLResponse)
async def layers_dashboard():
    """分层模型系统监控仪表盘"""
    return LAYER_DASHBOARD_HTML


class LayerInitRequest(BaseModel):
    models_per_layer: int = 5


class LayerClaimRequest(BaseModel):
    level: Optional[int] = None
    owner: Optional[str] = None
    use_pool: bool = False


class LayerPredictRequest(BaseModel):
    prompt: str = "Hello"
    level: Optional[int] = None


@app.get("/api/layers/stats")
async def layers_stats():
    """获取分层系统统计"""
    system = _get_layer_system()
    return system.statistics()


@app.get("/api/layers/list")
async def layers_list():
    """列出所有层和模型"""
    system = _get_layer_system()
    result = []
    for layer in system.get_layers_ordered():
        layer_info = {
            "level": layer.config.level,
            "layer_id": layer.config.layer_id,
            "layer_name": layer.config.layer_name,
            "layer_type": layer.config.layer_type.name,
            "models": [],
        }
        for model in layer.models.values():
            layer_info["models"].append({
                "model_id": model.model_id,
                "owner": model.owner,
                "training_state": model.training_state.name,
                "training_progress": round(model.training_progress, 2),
                "energy_buffer": round(model._energy_buffer, 2),
                "total_calls": model.stats.total_calls,
            })
        result.append(layer_info)
    return {"layers": result}


@app.post("/api/layers/init")
async def layers_init(req: LayerInitRequest):
    """初始化分层系统"""
    global _layer_system
    _layer_system = LayeredModelSystem()
    _layer_system.setup_default_layers(models_per_layer=req.models_per_layer)
    _layer_system.save(_layer_state_file)
    return {"status": "ok", "stats": _layer_system.statistics()}


@app.post("/api/layers/claim")
async def layers_claim(req: LayerClaimRequest):
    """认领模型"""
    system = _get_layer_system()
    if req.use_pool:
        assignments = system.auto_assign_from_pool()
        total = sum(len(a) for a in assignments.values())
        system.save(_layer_state_file)
        return {"status": "ok", "assigned": total, "assignments": assignments}
    elif req.level and req.owner:
        layer = system.get_layer_by_level(req.level)
        if layer:
            unclaimed = layer.get_unclaimed()
            if unclaimed:
                model = unclaimed[0]
                if model.claim(req.owner):
                    system.save(_layer_state_file)
                    return {"status": "ok", "model_id": model.model_id, "owner": req.owner}
        return {"status": "error", "message": "No unclaimed models or layer not found"}
    return {"status": "error", "message": "Provide (level+owner) or use_pool=true"}


@app.post("/api/layers/train")
async def layers_train():
    """训练所有已认领的模型"""
    system = _get_layer_system()
    for layer in system.get_layers_ordered():
        for model in layer.models.values():
            if model.training_state == TrainingState.CLAIMED:
                model.start_training()
    result = system.train_until_complete(step_progress=0.3, max_steps=10)
    system.save(_layer_state_file)
    return {"status": "ok", "result": result}


@app.post("/api/layers/predict")
async def layers_predict(req: LayerPredictRequest):
    """调用模型预测"""
    system = _get_layer_system()
    # 充能
    for layer in system.get_layers_ordered():
        for model in layer.models.values():
            if model.training_state == TrainingState.TRAINED:
                model.charge(model.energy_requirement * 3)

    test_input = ModelInput(prompt=req.prompt)
    if req.level:
        layer = system.get_layer_by_level(req.level)
        if layer:
            output = layer.ensemble_predict(test_input)
            if output:
                system.save(_layer_state_file)
                return {
                    "status": "ok",
                    "level": req.level,
                    "layer_name": layer.config.layer_name,
                    "classification": output.classification,
                    "prediction": output.prediction,
                    "text": output.text,
                    "json": output.json,
                }
            return {"status": "error", "message": "No trained models in this layer"}
        return {"status": "error", "message": "Layer not found"}
    else:
        results = system.ensemble_all_layers(test_input)
        system.save(_layer_state_file)
        return {"status": "ok", "results": results}


@app.post("/api/layers/save")
async def layers_save():
    """保存状态"""
    system = _get_layer_system()
    ok = system.save(_layer_state_file)
    return {"status": "ok" if ok else "error"}


@app.get("/api/layers/ai-pool")
async def layers_ai_pool():
    """获取可用 AI 名称池"""
    return {"ai_names": AI_NAME_POOL, "count": len(AI_NAME_POOL)}


# 分层系统仪表盘 HTML
LAYER_DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Xuni 分层模型监控</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, sans-serif; background: #0a0a0a; color: #e0e0e0; padding: 16px; }
h1 { text-align: center; color: #00ff88; margin-bottom: 16px; font-size: 1.4rem; }
.stats-bar { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 16px; justify-content: center; }
.stat-card { background: #1a1a2e; padding: 10px 16px; border-radius: 8px; border: 1px solid #333; min-width: 100px; text-align: center; }
.stat-card .value { font-size: 1.5rem; font-weight: bold; color: #00ff88; }
.stat-card .label { font-size: 0.75rem; color: #888; margin-top: 4px; }
.actions { text-align: center; margin-bottom: 16px; }
button { background: #00ff88; color: #000; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-weight: bold; margin: 0 4px; }
button:hover { background: #00cc66; }
button.danger { background: #ff4466; }
button.danger:hover { background: #cc3355; }
.layers-container { max-width: 900px; margin: 0 auto; }
.layer-card { background: #1a1a2e; border: 1px solid #333; border-radius: 8px; padding: 12px; margin-bottom: 12px; }
.layer-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
.layer-title { font-weight: bold; color: #00aaff; }
.layer-progress { font-size: 0.8rem; color: #888; }
.models-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 8px; }
.model-item { background: #0d0d1a; padding: 8px; border-radius: 4px; border-left: 3px solid #555; font-size: 0.75rem; }
.model-item.trained { border-left-color: #00ff88; }
.model-item.training { border-left-color: #ffaa00; }
.model-item.claimed { border-left-color: #00aaff; }
.model-id { font-weight: bold; color: #ccc; }
.model-owner { color: #00aaff; }
.model-state { font-size: 0.65rem; color: #888; }
.progress-bar { width: 100%; height: 4px; background: #333; border-radius: 2px; margin-top: 4px; }
.progress-fill { height: 100%; background: #00ff88; border-radius: 2px; transition: width 0.3s; }
#log { background: #0d0d1a; padding: 12px; border-radius: 8px; margin-top: 16px; max-height: 200px; overflow-y: auto; font-family: monospace; font-size: 0.75rem; }
.log-entry { margin-bottom: 4px; }
.log-entry.ok { color: #00ff88; }
.log-entry.error { color: #ff4466; }
.log-entry.info { color: #00aaff; }
</style>
</head>
<body>
<h1>Xuni 分层模型监控</h1>
<div class="stats-bar" id="statsBar">
  <div class="stat-card"><div class="value" id="stat-layers">-</div><div class="label">层</div></div>
  <div class="stat-card"><div class="value" id="stat-models">-</div><div class="label">模型</div></div>
  <div class="stat-card"><div class="value" id="stat-claimed">-</div><div class="label">已认领</div></div>
  <div class="stat-card"><div class="value" id="stat-trained">-</div><div class="label">已训练</div></div>
  <div class="stat-card"><div class="value" id="stat-calls">-</div><div class="label">总调用</div></div>
  <div class="stat-card"><div class="value" id="stat-owners">-</div><div class="label">AI数</div></div>
</div>
<div class="actions">
  <button onclick="initSystem()">初始化</button>
  <button onclick="claimAll()">认领全部</button>
  <button onclick="trainAll()">训练</button>
  <button onclick="refresh()">刷新</button>
</div>
<div class="layers-container" id="layersContainer">
  <p style="text-align:center;color:#888">加载中...</p>
</div>
<div id="log"></div>
<script>
function log(msg, type) {
  type = type || 'info';
  var logDiv = document.getElementById('log');
  var entry = document.createElement('div');
  entry.className = 'log-entry ' + type;
  entry.textContent = '[' + new Date().toLocaleTimeString() + '] ' + msg;
  logDiv.insertBefore(entry, logDiv.firstChild);
  while (logDiv.children.length > 20) logDiv.removeChild(logDiv.lastChild);
}
async function api(url, method, body) {
  method = method || 'GET';
  var opts = { method: method };
  if (body) { opts.headers = {'Content-Type':'application/json'}; opts.body = JSON.stringify(body); }
  var resp = await fetch(url, opts);
  return resp.json();
}
async function refresh() {
  try {
    var stats = await api('/api/layers/stats');
    document.getElementById('stat-layers').textContent = stats.total_layers;
    document.getElementById('stat-models').textContent = stats.total_models;
    document.getElementById('stat-claimed').textContent = stats.total_claimed;
    document.getElementById('stat-trained').textContent = stats.total_trained;
    document.getElementById('stat-calls').textContent = stats.total_calls;
    document.getElementById('stat-owners').textContent = stats.unique_owners;
    var list = await api('/api/layers/list');
    var container = document.getElementById('layersContainer');
    container.innerHTML = '';
    list.layers.forEach(function(layer) {
      var trained = layer.models.filter(function(m){return m.training_state==='TRAINED'}).length;
      var claimed = layer.models.filter(function(m){return m.owner}).length;
      var card = document.createElement('div');
      card.className = 'layer-card';
      var header = '<div class="layer-header"><div class="layer-title">L' + layer.level + ' ' + layer.layer_name + '</div><div class="layer-progress">' + claimed + '/' + layer.models.length + ' 认领 | ' + trained + ' 已训练</div></div>';
      var grid = '<div class="models-grid">';
      layer.models.forEach(function(m) {
        var cls = 'model-item ' + m.training_state.toLowerCase();
        var owner = m.owner ? '<div class="model-owner">@' + m.owner + '</div>' : '';
        var bar = m.training_progress > 0 ? '<div class="progress-bar"><div class="progress-fill" style="width:'+(m.training_progress*100)+'%"></div></div>' : '';
        grid += '<div class="' + cls + '"><div class="model-id">' + m.model_id + '</div>' + owner + '<div class="model-state">' + m.training_state + ' ' + (m.training_progress*100).toFixed(0) + '%</div>' + bar + '</div>';
      });
      grid += '</div>';
      card.innerHTML = header + grid;
      container.appendChild(card);
    });
  } catch(e) { log('刷新失败: ' + e.message, 'error'); }
}
async function initSystem() {
  log('初始化分层系统...');
  var r = await api('/api/layers/init', 'POST', {models_per_layer: 5});
  log('初始化完成: ' + r.stats.total_models + ' 个模型', 'ok');
  refresh();
}
async function claimAll() {
  log('认领所有模型...');
  var r = await api('/api/layers/claim', 'POST', {use_pool: true});
  log('认领完成: ' + r.assigned + ' 个模型', 'ok');
  refresh();
}
async function trainAll() {
  log('开始训练...');
  var r = await api('/api/layers/train', 'POST');
  log('训练完成: ' + r.result.final_trained + ' 个模型已训练', 'ok');
  refresh();
}
refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>
"""


# 开发服务器入口
def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
