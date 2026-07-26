"""
玻璃逻辑 pipeline 示例

将采样、场计算、音乐合成建模为光学系统。
"""

import numpy as np
from xuni.glass import XuniGlass, OpticalMedium
from xuni.sampler import XuniSampler, SamplingMode
from xuni.field import XuniField
from xuni.converter import XuniConverter
from xuni.music import XuniMusic

# 构建光学系统
glass = XuniGlass("music_pipeline")

# 元件1：棱镜 - 分离混沌的"频段"
glass.add_element(
    "sampler_prism",
    OpticalMedium.PRISM,
    dispersion=0.3,
    func=lambda _: XuniSampler(mode=SamplingMode.HYPER_CHAOS, seed=42).generate_batch(50000)
)

# 元件2：透镜 - 聚焦场能量
glass.add_element(
    "field_lens",
    OpticalMedium.LENS,
    focus=0.2,
    func=lambda batch: _compute_field(batch)
)

# 元件3：玻璃 - 参数转换
glass.add_element(
    "converter_glass",
    OpticalMedium.GLASS,
    n=1.8,
    func=lambda field_data: _convert(field_data)
)

# 元件4：晶体 - 双路径音乐合成
glass.add_element(
    "music_crystal",
    OpticalMedium.CRYSTAL,
    func=lambda params: _synthesize(params)
)

def _compute_field(batch):
    field = XuniField(grid_size=(16, 16, 16))
    field.ingest_batch(batch)
    field.compute_field()
    return field

def _convert(field):
    converter = XuniConverter()
    summary = field.field_summary()
    summary["dominant_ex"] = field.get_dominant_vector()[0]
    summary["dominant_ey"] = field.get_dominant_vector()[1]
    summary["dominant_ez"] = field.get_dominant_vector()[2]
    return converter.convert(summary, field.get_energy_distribution())

def _synthesize(params):
    music = XuniMusic(sample_rate=22050)
    return music.synthesize(params, duration=3.0)

# 发射一束光
print("Shining light through the pipeline...")
ray = glass.shine(None)
report = glass.get_optical_report()

print(f"\nOptical report:")
print(f"  Path: {' -> '.join(report['elements'])}")
print(f"  Intensity: {report['latest_ray']['intensity']:.4f}")
print(f"  Refractions: {report['latest_ray']['refractions']}")

if isinstance(ray.payload, dict) and "error" in ray.payload:
    print(f"Error: {ray.payload['error']}")
else:
    audio = ray.payload
    wav = XuniMusic.to_wav_bytes(audio)
    with open("xuni_glass.wav", "wb") as f:
        f.write(wav)
    print("Saved to xuni_glass.wav")

# 共振演示
print("\nResonance loop demo:")
rays = glass.resonance_loop(None, iterations=3, feedback_gain=0.2)
for i, r in enumerate(rays):
    print(f"  Iteration {i+1}: intensity={r.intensity:.4f}, path={r.get_path_length()}")
