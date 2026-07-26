"""
水动力学采样 → 音乐

使用流体粒子代替混沌系统作为采样源。
"""

from xuni.hydro import XuniHydro
from xuni.field import XuniField
from xuni.converter import XuniConverter
from xuni.music import XuniMusic

# 1. 运行水动力学模拟
hydro = XuniHydro(n_particles=4096, seed=42, viscosity=0.15)
print("Running hydrodynamics...")
batch = hydro.get_sample_batch(50000)
print(f"Fluid particles sampled: {len(batch)}")

# 2. 构建电场
field = XuniField(grid_size=(24, 24, 24))
field.ingest_batch(batch)
field.compute_field()

# 3. 转换与合成
converter = XuniConverter()
params = converter.convert(field.field_summary(), field.get_energy_distribution())
music = XuniMusic(sample_rate=22050)
audio = music.synthesize(params, duration=8.0)

wav = music.to_wav_bytes(audio)
with open("xuni_hydro.wav", "wb") as f:
    f.write(wav)
print("Saved to xuni_hydro.wav")
