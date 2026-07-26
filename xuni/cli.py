"""
XuniCLI —— 命令行接口

提供终端命令直接操作虚拟电场与音乐合成。
"""

import argparse
import sys
import numpy as np

from .sampler import XuniSampler, SamplingMode
from .field import XuniField
from .converter import XuniConverter
from .music import XuniMusic, AudioBuffer
from .hydro import XuniHydro
from .glass import XuniGlass, OpticalMedium
from .brain import XuniBrain
from .trainer import XuniTrainer, TrainingConfig
from .memory import XuniMemory
from .critic import XuniCritic
from .explorer import XuniExplorer, SamplingStrategy
from .credential import XuniCredential, CredentialType, TokenStatus
from .model import XuniModelRegistry, ModelInput
from .gateway import XuniGateway, APIEndpoint, APIRequest
from .layer import LayeredModelSystem, LayerType, LayerConfig, AI_NAME_POOL
from .model import TrainingState


def cmd_sample(args):
    """生成采样点"""
    mode_map = {
        "chaos": SamplingMode.HYPER_CHAOS,
        "lorenz": SamplingMode.LORENZ_96,
        "mandelbulb": SamplingMode.MANDELBULB,
        "noise": SamplingMode.NOISE_FIELD,
        "hybrid": SamplingMode.HYBRID,
    }
    mode = mode_map.get(args.mode, SamplingMode.HYPER_CHAOS)
    sampler = XuniSampler(mode=mode, seed=args.seed)
    batch = sampler.generate_batch(args.count)
    print(f"Generated {args.count} samples")
    print(f"  X range: [{batch[:,0].min():.3f}, {batch[:,0].max():.3f}]")
    print(f"  Y range: [{batch[:,1].min():.3f}, {batch[:,1].max():.3f}]")
    print(f"  Z range: [{batch[:,2].min():.3f}, {batch[:,2].max():.3f}]")
    print(f"  Mean entropy: {batch[:,5].mean():.4f}")
    if args.output:
        np.save(args.output, batch)
        print(f"Saved to {args.output}")


def cmd_field(args):
    """计算虚拟电场"""
    mode_map = {
        "chaos": SamplingMode.HYPER_CHAOS,
        "lorenz": SamplingMode.LORENZ_96,
        "mandelbulb": SamplingMode.MANDELBULB,
        "noise": SamplingMode.NOISE_FIELD,
        "hybrid": SamplingMode.HYBRID,
        "hydro": None,  # special
    }

    if args.mode == "hydro":
        hydro = XuniHydro(n_particles=4096, seed=args.seed)
        batch = hydro.get_sample_batch(args.count)
    else:
        mode = mode_map[args.mode]
        sampler = XuniSampler(mode=mode, seed=args.seed)
        batch = sampler.generate_batch(args.count)

    field = XuniField(grid_size=(args.grid, args.grid, args.grid))
    field.ingest_batch(batch)
    field.compute_field()
    summary = field.field_summary()
    print("Field Summary:")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4e}")
        else:
            print(f"  {k}: {v}")


def cmd_music(args):
    """合成音乐"""
    mode_map = {
        "chaos": SamplingMode.HYPER_CHAOS,
        "lorenz": SamplingMode.LORENZ_96,
        "mandelbulb": SamplingMode.MANDELBULB,
        "noise": SamplingMode.NOISE_FIELD,
        "hybrid": SamplingMode.HYBRID,
        "hydro": None,
    }

    if args.mode == "hydro":
        hydro = XuniHydro(n_particles=4096, seed=args.seed)
        batch = hydro.get_sample_batch(args.count)
    else:
        mode = mode_map[args.mode]
        sampler = XuniSampler(mode=mode, seed=args.seed)
        batch = sampler.generate_batch(args.count)

    field = XuniField(grid_size=(16, 16, 16))
    field.ingest_batch(batch)
    field.compute_field()

    summary = field.field_summary()
    summary["dominant_ex"] = field.get_dominant_vector()[0]
    summary["dominant_ey"] = field.get_dominant_vector()[1]
    summary["dominant_ez"] = field.get_dominant_vector()[2]

    converter = XuniConverter()
    params = converter.convert(summary, field.get_energy_distribution())

    music = XuniMusic(sample_rate=args.rate)
    audio = music.synthesize(params, duration=args.duration)

    if args.output:
        wav = music.to_wav_bytes(audio)
        with open(args.output, "wb") as f:
            f.write(wav)
        print(f"Saved {args.duration}s audio to {args.output}")
        print(f"  Frequency: {params.base_frequency:.2f} Hz")
        print(f"  Tempo: {params.tempo:.1f} BPM")
        print(f"  Harmonics: {params.harmonics}")
    else:
        print(f"Generated {args.duration}s audio (no output file)")
        print(f"  Frequency: {params.base_frequency:.2f} Hz")
        print(f"  Tempo: {params.tempo:.1f} BPM")


def cmd_hydro(args):
    """运行水动力学模拟"""
    hydro = XuniHydro(
        n_particles=args.particles,
        seed=args.seed,
        viscosity=args.viscosity,
    )
    for step in range(args.steps):
        hydro._step()
        if step % 50 == 0:
            summary = hydro.hydro_summary()
            print(f"Step {step}: {summary.get('particle_count', 0)} particles, "
                  f"T_mean={summary.get('temperature_mean', 0):.1f}, "
                  f"vort={summary.get('vorticity_mean', 0):.4f}")


def cmd_glass(args):
    """运行玻璃逻辑演示"""
    glass = XuniGlass("demo")

    # 构建光学系统
    glass.add_element("input_lens", OpticalMedium.LENS, focus=0.3, n=1.2)
    glass.add_element("prism_split", OpticalMedium.PRISM, dispersion=0.5, n=1.5)
    glass.add_element("field_transform", OpticalMedium.GLASS,
                      func=lambda x: np.tanh(x) if isinstance(x, np.ndarray) else x,
                      n=2.0)
    glass.add_element("output_mirror", OpticalMedium.MIRROR, reflectivity=0.2)

    data = np.random.standard_normal(100)
    ray = glass.shine(data, wavelength=550.0)

    report = glass.get_optical_report()
    print("Optical Report:")
    print(f"  System: {report['system_name']}")
    print(f"  Elements: {report['elements']}")
    print(f"  Path length: {report['latest_ray']['path_length']}")
    print(f"  Intensity: {report['latest_ray']['intensity']:.4f}")
    print(f"  Refractions: {report['latest_ray']['refractions']}")

    if args.resonance:
        print("\nResonance loop:")
        rays = glass.resonance_loop(data, iterations=3, feedback_gain=0.3)
        for i, r in enumerate(rays):
            print(f"  Iteration {i+1}: intensity={r.intensity:.4f}")


def cmd_brain(args):
    """场能量驱动 Brain 生成音乐"""
    mode_map = {
        "chaos": SamplingMode.HYPER_CHAOS,
        "lorenz": SamplingMode.LORENZ_96,
        "mandelbulb": SamplingMode.MANDELBULB,
        "noise": SamplingMode.NOISE_FIELD,
        "hybrid": SamplingMode.HYBRID,
    }
    mode = mode_map.get(args.mode, SamplingMode.HYPER_CHAOS)
    sampler = XuniSampler(mode=mode, seed=args.seed)
    field = XuniField(grid_size=(16, 16, 16))

    # 生成场
    batch = sampler.generate_batch(args.count)
    field.ingest_batch(batch)
    field.compute_field()
    summary = field.field_summary()
    total_energy = summary.get("total_energy", 1.0)
    field_energy = np.log1p(total_energy)

    # 创建 Brain
    brain = XuniBrain(
        n_neurons=args.neurons,
        sample_rate=args.rate,
        field_coupling=args.coupling,
        seed=args.seed,
    )

    if args.cultivate:
        # 培养模式：生成目标音频并培养
        converter = XuniConverter()
        music = XuniMusic(sample_rate=args.rate)
        summary["dominant_ex"] = field.get_dominant_vector()[0]
        summary["dominant_ey"] = field.get_dominant_vector()[1]
        summary["dominant_ez"] = field.get_dominant_vector()[2]
        params = converter.convert(summary, field.get_energy_distribution())
        target_audio = music.synthesize(params, duration=args.duration).to_mono()

        trainer = XuniTrainer(brain, config=TrainingConfig())
        print(f"Cultivating brain with field_energy={field_energy:.4f} for {args.epochs} epoch(s)...")
        output = trainer.cultivate(
            target_audio=target_audio,
            duration=args.duration,
            field_energy=field_energy,
            epochs=args.epochs,
        )
        print("Cultivation complete.")
        audio = output
    else:
        # 直接共振生成
        print(f"Stimulating brain with field_energy={field_energy:.4f}...")
        audio = brain.stimulate(duration=args.duration, field_energy=field_energy)

    # 保存
    music = XuniMusic(sample_rate=args.rate)
    buf = AudioBuffer(sample_rate=args.rate, data=audio, duration=args.duration)
    wav = music.to_wav_bytes(buf)
    with open(args.output, "wb") as f:
        f.write(wav)
    print(f"Saved {args.duration}s brain audio to {args.output}")
    print(f"  Neurons: {brain.n}, Field coupling: {brain.field_coupling}")
    print(f"  Sync: {brain.brain_summary()['synchronization']:.4f}")


def cmd_critic(args):
    """评估音乐质量（四维认知不变量）"""
    mode_map = {
        "chaos": SamplingMode.HYPER_CHAOS,
        "lorenz": SamplingMode.LORENZ_96,
        "mandelbulb": SamplingMode.MANDELBULB,
        "noise": SamplingMode.NOISE_FIELD,
        "hybrid": SamplingMode.HYBRID,
    }
    mode = mode_map.get(args.mode, SamplingMode.HYPER_CHAOS)
    sampler = XuniSampler(mode=mode, seed=args.seed)
    field = XuniField(grid_size=(16, 16, 16))
    converter = XuniConverter()
    music = XuniMusic(sample_rate=args.rate)

    batch = sampler.generate_batch(args.count)
    field.ingest_batch(batch)
    field.compute_field()
    summary = field.field_summary()
    summary["dominant_ex"] = field.get_dominant_vector()[0]
    summary["dominant_ey"] = field.get_dominant_vector()[1]
    summary["dominant_ez"] = field.get_dominant_vector()[2]
    params = converter.convert(summary, field.get_energy_distribution())
    audio = music.synthesize(params, duration=args.duration).to_mono()

    critic = XuniCritic(sample_rate=args.rate)
    scores = critic.evaluate(audio)
    suggestions = critic.suggest_optimization(scores)

    print("Music Invariant Scores:")
    for k, v in scores.to_dict().items():
        print(f"  {k.upper()}: {v}")
    print("Suggestions:")
    for k, v in suggestions.items():
        print(f"  [{k}] {v}")


def cmd_explore(args):
    """探索-利用循环，自动寻找最佳采样模式"""
    explorer = XuniExplorer(epsilon=0.4)
    for mode in ["hyper_chaos", "lorenz_96", "mandelbulb", "noise_field", "hybrid"]:
        explorer.register_sampling_mode(SamplingStrategy(mode))

    music = XuniMusic(sample_rate=22050)
    converter = XuniConverter()
    critic = XuniCritic(sample_rate=22050)

    print(f"Running {args.trials} exploration trials...")
    for trial in range(args.trials):
        name, params = explorer.select_strategy(category="sample")
        mode_str = params["mode"]
        mode_map = {
            "hyper_chaos": SamplingMode.HYPER_CHAOS,
            "lorenz_96": SamplingMode.LORENZ_96,
            "mandelbulb": SamplingMode.MANDELBULB,
            "noise_field": SamplingMode.NOISE_FIELD,
            "hybrid": SamplingMode.HYBRID,
        }
        mode = mode_map[mode_str]
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
        audio = music.synthesize(mp, duration=args.duration).to_mono()
        scores = critic.evaluate(audio)
        explorer.feedback(name, scores.overall)
        print(f"  Trial {trial+1}: {mode_str} -> overall={scores.overall:.4f}")

    report = explorer.get_report()
    print("\nExplorer Report:")
    print(f"  Epsilon: {report['epsilon']:.4f}")
    for s in report["strategies"]:
        print(f"  {s['name']}: avg={s['avg_score']:.4f}, trials={s['trials']}, novelty={s['novelty']}")


def main():
    parser = argparse.ArgumentParser(description="Xuni Virtual Field & Music System")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # sample
    p_sample = subparsers.add_parser("sample", help="Generate sampling points")
    p_sample.add_argument("--mode", default="chaos", choices=["chaos","lorenz","mandelbulb","noise","hybrid"])
    p_sample.add_argument("--count", type=int, default=10000)
    p_sample.add_argument("--seed", type=int, default=42)
    p_sample.add_argument("--output", default=None, help="Output .npy file")
    p_sample.set_defaults(func=cmd_sample)

    # field
    p_field = subparsers.add_parser("field", help="Compute virtual electric field")
    p_field.add_argument("--mode", default="chaos", choices=["chaos","lorenz","mandelbulb","noise","hybrid","hydro"])
    p_field.add_argument("--count", type=int, default=100000)
    p_field.add_argument("--grid", type=int, default=32)
    p_field.add_argument("--seed", type=int, default=42)
    p_field.set_defaults(func=cmd_field)

    # music
    p_music = subparsers.add_parser("music", help="Synthesize music")
    p_music.add_argument("--mode", default="chaos", choices=["chaos","lorenz","mandelbulb","noise","hybrid","hydro"])
    p_music.add_argument("--count", type=int, default=100000)
    p_music.add_argument("--duration", type=float, default=5.0)
    p_music.add_argument("--rate", type=int, default=22050)
    p_music.add_argument("--seed", type=int, default=42)
    p_music.add_argument("--output", default="xuni_output.wav")
    p_music.set_defaults(func=cmd_music)

    # hydro
    p_hydro = subparsers.add_parser("hydro", help="Run hydrodynamics simulation")
    p_hydro.add_argument("--particles", type=int, default=4096)
    p_hydro.add_argument("--steps", type=int, default=200)
    p_hydro.add_argument("--viscosity", type=float, default=0.1)
    p_hydro.add_argument("--seed", type=int, default=42)
    p_hydro.set_defaults(func=cmd_hydro)

    # glass
    p_glass = subparsers.add_parser("glass", help="Run glass logic demo")
    p_glass.add_argument("--resonance", action="store_true", help="Enable resonance loop")
    p_glass.set_defaults(func=cmd_glass)

    # brain
    p_brain = subparsers.add_parser("brain", help="Field-driven resonant brain music")
    p_brain.add_argument("--mode", default="chaos", choices=["chaos","lorenz","mandelbulb","noise","hybrid"])
    p_brain.add_argument("--count", type=int, default=100000)
    p_brain.add_argument("--duration", type=float, default=5.0)
    p_brain.add_argument("--rate", type=int, default=22050)
    p_brain.add_argument("--seed", type=int, default=42)
    p_brain.add_argument("--neurons", type=int, default=256)
    p_brain.add_argument("--coupling", type=float, default=0.5)
    p_brain.add_argument("--cultivate", action="store_true", help="Enable cultivation mode")
    p_brain.add_argument("--epochs", type=int, default=1)
    p_brain.add_argument("--output", default="xuni_brain.wav")
    p_brain.set_defaults(func=cmd_brain)

    # critic
    p_critic = subparsers.add_parser("critic", help="Evaluate music with cognitive invariants")
    p_critic.add_argument("--mode", default="chaos", choices=["chaos","lorenz","mandelbulb","noise","hybrid"])
    p_critic.add_argument("--count", type=int, default=100000)
    p_critic.add_argument("--duration", type=float, default=3.0)
    p_critic.add_argument("--rate", type=int, default=22050)
    p_critic.add_argument("--seed", type=int, default=42)
    p_critic.set_defaults(func=cmd_critic)

    # explore
    p_explore = subparsers.add_parser("explore", help="Explore-exploit best sampling mode")
    p_explore.add_argument("--trials", type=int, default=5)
    p_explore.add_argument("--duration", type=float, default=3.0)
    p_explore.set_defaults(func=cmd_explore)

    # credential
    p_cred = subparsers.add_parser("credential", help="Virtual credential management")
    p_cred.add_argument("--action", default="mint", choices=["mint", "list", "validate", "stats"])
    p_cred.add_argument("--energy", type=float, default=10.0)
    p_cred.add_argument("--type", default="access", choices=["access", "model", "premium", "api"])
    p_cred.add_argument("--token-id", default=None)
    p_cred.set_defaults(func=cmd_credential)

    # model
    p_model = subparsers.add_parser("model", help="Virtual model operations")
    p_model.add_argument("--action", default="list", choices=["list", "info", "predict"])
    p_model.add_argument("--model-id", default=None)
    p_model.add_argument("--prompt", default="")
    p_model.add_argument("--energy", type=float, default=100.0)
    p_model.set_defaults(func=cmd_model)

    # ecosystem
    p_eco = subparsers.add_parser("ecosystem", help="Run virtual ecosystem demo")
    p_eco.add_argument("--samples", type=int, default=10000)
    p_eco.add_argument("--seed", type=int, default=42)
    p_eco.set_defaults(func=cmd_ecosystem)

    # layer
    p_layer = subparsers.add_parser("layer", help="Layered model system operations")
    p_layer.add_argument("--action", default="list",
                         choices=["list", "init", "claim", "train", "predict", "save", "load", "stats", "viz"])
    p_layer.add_argument("--file", default="xuni_layers.json", help="JSON file for save/load")
    p_layer.add_argument("--level", type=int, default=None, help="Layer level")
    p_layer.add_argument("--model-id", default=None)
    p_layer.add_argument("--owner", default=None, help="AI name to claim")
    p_layer.add_argument("--prompt", default="Hello")
    p_layer.add_argument("--count", type=int, default=5, help="Models per layer")
    p_layer.add_argument("--pool", action="store_true", help="Auto assign from AI pool")
    p_layer.set_defaults(func=cmd_layer)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    args.func(args)


def _load_or_init_system(filepath: str = "xuni_layers.json", count: int = 5) -> LayeredModelSystem:
    """加载已有系统或新建"""
    system = LayeredModelSystem.load(filepath)
    if system is None:
        system = LayeredModelSystem()
        system.setup_default_layers(models_per_layer=count)
    return system


def cmd_layer(args):
    """分层模型系统操作"""
    if args.action == "init":
        system = LayeredModelSystem()
        system.setup_default_layers(models_per_layer=args.count)
        system.save(args.file)
        print(f"Initialized {system.statistics()['total_models']} models in {system.statistics()['total_layers']} layers")
        print(f"Saved to {args.file}")
        return

    if args.action == "load":
        system = LayeredModelSystem.load(args.file)
        if system is None:
            print(f"Failed to load from {args.file}")
            return
        print(f"Loaded from {args.file}")
        print(system.visualize())
        return

    # 其他操作需要加载系统
    system = _load_or_init_system(args.file, args.count)

    if args.action == "list":
        print(system.visualize())

    elif args.action == "stats":
        stats = system.statistics()
        print(f"Layers: {stats['total_layers']}")
        print(f"Models: {stats['total_models']}")
        print(f"Claimed: {stats['total_claimed']}")
        print(f"Trained: {stats['total_trained']}")
        print(f"Total calls: {stats['total_calls']}")
        print(f"Total energy: {stats['total_energy_consumed']}")
        print(f"Owners ({stats['unique_owners']}): {', '.join(stats['owners'])}")
        for ls in stats["layers"]:
            print(f"  L{ls['level']} {ls['layer_name']}: {ls['claimed']}/{ls['total_models']} claimed, {ls['trained']} trained")

    elif args.action == "claim":
        if args.pool:
            assignments = system.auto_assign_from_pool()
            total = sum(len(a) for a in assignments.values())
            print(f"Auto-assigned {total} models from AI pool")
        elif args.level and args.owner:
            layer = system.get_layer_by_level(args.level)
            if layer:
                unclaimed = layer.get_unclaimed()
                if unclaimed:
                    model = unclaimed[0]
                    if model.claim(args.owner):
                        print(f"{args.owner} claimed {model.model_id} (Layer {args.level})")
                else:
                    print(f"No unclaimed models in Layer {args.level}")
        else:
            print("Usage: --claim --level N --owner NAME  OR  --claim --pool")
        system.save(args.file)

    elif args.action == "train":
        # 开始训练所有已认领的
        for layer in system.get_layers_ordered():
            for model in layer.models.values():
                if model.training_state == TrainingState.CLAIMED:
                    model.start_training()
        # 协作训练到完成
        result = system.train_until_complete(step_progress=0.3, max_steps=10)
        print(f"Training completed in {result['total_steps']} steps")
        for h in result["history"]:
            print(f"  Step {h['step']}: {h['trained']}/{h['claimed']} trained")
        system.save(args.file)

    elif args.action == "predict":
        # 充能
        for layer in system.get_layers_ordered():
            for model in layer.models.values():
                if model.training_state == TrainingState.TRAINED:
                    model.charge(model.energy_requirement * 3)

        test_input = ModelInput(prompt=args.prompt)
        if args.level:
            layer = system.get_layer_by_level(args.level)
            if layer:
                output = layer.ensemble_predict(test_input)
                if output:
                    print(f"Layer {args.level} ({layer.config.layer_name}) ensemble result:")
                    if output.classification:
                        print(f"  Classification: {output.classification}")
                    if output.prediction is not None:
                        print(f"  Prediction: {output.prediction:.2f}")
                    if output.text:
                        print(f"  Text: {output.text[:100]}")
                    if output.json:
                        print(f"  Data: {output.json}")
                else:
                    print("No trained models in this layer")
        else:
            results = system.ensemble_all_layers(test_input)
            for layer_id, r in results.items():
                print(f"\nLayer {r['level']} ({r['layer_name']}):")
                if r.get("classification"):
                    print(f"  Classification: {r['classification']}")
                if r.get("prediction") is not None:
                    print(f"  Prediction: {r['prediction']:.2f}")
                if r.get("text"):
                    print(f"  Text: {r['text'][:100]}")
                if r.get("json"):
                    print(f"  Data: {r['json']}")

    elif args.action == "save":
        system.save(args.file)
        print(f"Saved to {args.file}")

    elif args.action == "viz":
        print(system.visualize())


def cmd_credential(args):
    """虚拟凭证管理"""
    credential = XuniCredential()
    
    if args.action == "mint":
        type_map = {
            "access": CredentialType.ACCESS_TOKEN,
            "model": CredentialType.MODEL_TOKEN,
            "premium": CredentialType.PREMIUM_TOKEN,
            "api": CredentialType.API_KEY,
        }
        token = credential.mint(
            field_energy=args.energy,
            token_type=type_map[args.type],
        )
        print(f"Token ID: {token.token_id}")
        print(f"Type: {token.token_type.name}")
        print(f"Energy Value: {token.energy_value:.2f}")
        print(f"JWT: {token.to_jwt()[:60]}...")
    
    elif args.action == "list":
        tokens = credential.list_tokens()
        print(f"Total tokens: {len(tokens)}")
        for t in tokens:
            print(f"  {t['token_id'][:12]}... [{t['type']}] {t['status']}")
    
    elif args.action == "validate":
        if not args.token_id:
            print("Error: --token-id is required for validate")
            return
        token = credential.validate(args.token_id)
        if token:
            print(f"Valid token: {token.token_type.name}")
        else:
            print("Invalid or expired token")
    
    elif args.action == "stats":
        stats = credential.statistics()
        print("Credential Statistics:")
        for k, v in stats.items():
            print(f"  {k}: {v}")


def cmd_model(args):
    """虚拟模型操作"""
    model_registry = XuniModelRegistry()
    model_registry.register_default_models()
    
    if args.action == "list":
        models = model_registry.list_models()
        print(f"Registered models: {len(models)}")
        for m in models:
            print(f"  {m['model_id']} - {m['model_type']}")
    
    elif args.action == "info":
        if not args.model_id:
            print("Error: --model-id is required")
            return
        model = model_registry.get_model(args.model_id)
        if model:
            info = model.get_info()
            print(f"Model: {info['model_id']}")
            print(f"Type: {info['model_type']}")
            print(f"Energy Requirement: {info['energy_requirement']}")
            print(f"Status: {info['status']}")
        else:
            print(f"Model not found: {args.model_id}")
    
    elif args.action == "predict":
        if not args.model_id:
            print("Error: --model-id is required")
            return
        model = model_registry.get_model(args.model_id)
        if not model:
            print(f"Model not found: {args.model_id}")
            return
        
        model.charge(args.energy)
        output = model.predict(ModelInput(prompt=args.prompt))
        
        if output.text:
            print(f"Text Output: {output.text}")
        if output.json:
            import json
            print(f"JSON Output:\n{json.dumps(output.json, indent=2)}")
        if output.classification:
            print(f"Classification: {output.classification}")
        print(f"Energy Consumed: {output.energy_consumed}")
        print(f"Latency: {output.latency_ms:.2f}ms")


def cmd_ecosystem(args):
    """运行虚拟生态系统"""
    from examples.virtual_ecosystem import run_virtual_ecosystem
    run_virtual_ecosystem()


if __name__ == "__main__":
    main()
