"""
基础用法示例：从采样点到音乐
"""

from xuni.sampler import XuniSampler, SamplingMode
from xuni.field import XuniField
from xuni.converter import XuniConverter
from xuni.music import XuniMusic

# 1. 生成采样点
sampler = XuniSampler(mode=SamplingMode.HYPER_CHAOS, seed=42)
batch = sampler.generate_batch(100000)
print(f"Generated {len(batch)} samples")

# 2. 构建虚拟电场
field = XuniField(grid_size=(32, 32, 32))
field.ingest_batch(batch)
field.compute_field()
summary = field.field_summary()
print(f"Field total energy: {summary['total_energy']:.4e}")

# 3. 转换为音乐参数
converter = XuniConverter()
params = converter.convert(summary, field.get_energy_distribution())
print(f"Base frequency: {params.base_frequency:.2f} Hz")
print(f"Tempo: {params.tempo:.1f} BPM")

# 4. 合成音乐
music = XuniMusic(sample_rate=22050)
audio = music.synthesize(params, duration=5.0)
wav = music.to_wav_bytes(audio)
with open("xuni_basic.wav", "wb") as f:
    f.write(wav)
print("Saved to xuni_basic.wav")
