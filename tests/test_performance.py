"""
性能测试脚本——验证生产速度优化效果

测试维度：
1. 采样器批量生成速度
2. 噪声场计算速度
3. 场计算速度
4. 资源工厂批量生产速度
5. 虚拟算力转换速度

优化目标：各模块速度提升 1000+ 倍
"""

import time
import numpy as np
from xuni.sampler import XuniSampler, SamplingMode
from xuni.field import XuniField
from xuni.multiverse_resources import MultiverseResourceFactory
from xuni.virtual_compute import VirtualComputeUnit


def test_sampler_performance():
    """测试采样器性能"""
    print("=" * 60)
    print("测试 1: 采样器批量生成速度")
    print("=" * 60)
    
    sampler = XuniSampler(mode=SamplingMode.HYPER_CHAOS, seed=42)
    
    for size in [1000, 10000, 100000, 1000000]:
        start = time.time()
        batch = sampler.generate_batch(batch_size=size)
        elapsed = time.time() - start
        
        rate = size / elapsed
        print(f"  生成 {size:,} 个采样点: {elapsed:.4f}s, 速度: {rate:,.0f}/s")
        
        assert batch.shape == (size, 6), f"形状错误: {batch.shape}"
        assert np.all(np.isfinite(batch)), "包含非有限值"
    
    print()


def test_noise_performance():
    """测试噪声计算性能"""
    print("=" * 60)
    print("测试 2: 噪声场计算速度（向量化版本）")
    print("=" * 60)
    
    sampler = XuniSampler(mode=SamplingMode.NOISE_FIELD, seed=42)
    
    for size in [1000, 10000, 100000, 500000]:
        start = time.time()
        batch = sampler.generate_batch(batch_size=size)
        elapsed = time.time() - start
        
        rate = size / elapsed
        print(f"  噪声计算 {size:,} 个点: {elapsed:.4f}s, 速度: {rate:,.0f}/s")
    
    print()


def test_field_performance():
    """测试场计算性能"""
    print("=" * 60)
    print("测试 3: 场计算速度（Gauss-Seidel 加速版）")
    print("=" * 60)
    
    sampler = XuniSampler(mode=SamplingMode.HYPER_CHAOS, seed=42)
    field = XuniField(grid_size=(32, 32, 32))
    
    # 准备数据
    batch = sampler.generate_batch(10000)
    
    for iterations in [20, 50, 100]:
        field.reset()
        field.ingest_batch(batch)
        
        start = time.time()
        field.compute_field(iterations=iterations, method="gauss_seidel")
        elapsed = time.time() - start
        
        energy = field.get_total_energy()
        print(f"  {iterations}次迭代: {elapsed:.4f}s, 总能量: {energy:.2f}")
    
    print()


def test_factory_performance():
    """测试资源工厂性能"""
    print("=" * 60)
    print("测试 4: 资源工厂批量生产速度")
    print("=" * 60)
    
    factory = MultiverseResourceFactory(parallel_lines=8, production_speed=10.0)
    
    blueprint = {
        "take": {"amount": 1000, "count": 10},
        "compression": {"factor": 50, "count": 5},
        "compute_core": {"density": 1e12, "count": 2},
    }
    
    for _ in range(5):
        start = time.time()
        resources = factory.mass_produce(blueprint)
        elapsed = time.time() - start
        
        rate = len(resources) / elapsed
        print(f"  生产 {len(resources)} 个资源: {elapsed:.4f}s, 速度: {rate:,.0f}/s")
    
    print()
    
    # 千万级结构化数组测试
    print("=" * 60)
    print("测试 4b: 资源工厂千万级结构化数组生产速度")
    print("=" * 60)
    
    factory2 = MultiverseResourceFactory(parallel_lines=1, production_speed=1.0)
    
    array_tests = [
        ("Take额度", lambda: factory2.produce_take_array(count=10000000)),
        ("虚拟流量", lambda: factory2.produce_bandwidth_array(count=10000000)),
        ("压缩点", lambda: factory2.produce_compression_array(count=10000000)),
        ("算力核心", lambda: factory2.produce_compute_core_array(count=10000000)),
        ("下载令牌", lambda: factory2.produce_download_token_array(count=10000000)),
        ("训练加速器", lambda: factory2.produce_training_accelerator_array(count=10000000)),
    ]
    
    for name, func in array_tests:
        start = time.time()
        arr = func()
        elapsed = time.time() - start
        rate = len(arr) / elapsed
        status = "✓" if rate >= 1_000_000 else "✗"
        print(f"  {status} {name}: {len(arr):,} 个, {elapsed:.4f}s, {rate:,.0f}/s")
    
    print()


def test_compute_performance():
    """测试虚拟算力转换性能"""
    print("=" * 60)
    print("测试 5: 虚拟算力转换速度")
    print("=" * 60)
    
    vcu = VirtualComputeUnit("VCU-TEST")
    
    # 单个注入
    for energy in [100, 1000, 10000]:
        start = time.time()
        for _ in range(100):
            vcu.inject_energy(energy)
        elapsed = time.time() - start
        
        rate = 100 / elapsed
        print(f"  单次注入 {energy} 电 × 100次: {elapsed:.4f}s, 速度: {rate:,.0f}/s")
    
    vcu2 = VirtualComputeUnit("VCU-BATCH")
    
    # 批量注入
    for count in [100, 1000, 10000]:
        energies = np.random.random(count) * 1000
        start = time.time()
        vcu2.inject_energy_batch(energies)
        elapsed = time.time() - start
        
        rate = count / elapsed
        print(f"  批量注入 {count} 次: {elapsed:.4f}s, 速度: {rate:,.0f}/s")
    
    print()


def test_mega_batch_performance():
    """测试超大规模批量生成"""
    print("=" * 60)
    print("测试 6: 超大规模批量生成（百万级）")
    print("=" * 60)
    
    sampler = XuniSampler(mode=SamplingMode.HYPER_CHAOS, seed=42)
    
    start = time.time()
    batch = sampler.generate_mega_batch(1000000)
    elapsed = time.time() - start
    
    rate = len(batch) / elapsed
    print(f"  生成 1,000,000 个采样点: {elapsed:.4f}s")
    print(f"  速度: {rate:,.0f}/s")
    print(f"  内存占用: {batch.nbytes / 1e6:.2f} MB")
    
    assert len(batch) == 1000000, f"生成数量错误: {len(batch)}"
    
    print()


def test_end_to_end_performance():
    """测试端到端性能：采样→场计算→产电→转换"""
    print("=" * 60)
    print("测试 7: 端到端性能（完整闭环）")
    print("=" * 60)
    
    sampler = XuniSampler(mode=SamplingMode.HYPER_CHAOS, seed=42)
    field = XuniField(grid_size=(32, 32, 32))
    vcu = VirtualComputeUnit("VCU-E2E")
    
    iterations = 10
    total_time = 0
    
    for i in range(iterations):
        start = time.time()
        
        batch = sampler.generate_batch(10000)
        field.reset()
        field.ingest_batch(batch)
        field.compute_field_fast()
        
        energy = field.get_total_energy()
        vcu.inject_energy(energy)
        
        elapsed = time.time() - start
        total_time += elapsed
        
        print(f"  迭代 {i+1}: {elapsed:.4f}s, 产电量: {energy:.2f}")
    
    avg_time = total_time / iterations
    print(f"\n  平均每次闭环: {avg_time:.4f}s")
    print(f"  每秒闭环次数: {1/avg_time:.1f}")
    
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("Xuni 性能测试套件")
    print("=" * 60)
    print()
    
    test_sampler_performance()
    test_noise_performance()
    test_field_performance()
    test_factory_performance()
    test_compute_performance()
    test_mega_batch_performance()
    test_end_to_end_performance()
    
    print("=" * 60)
    print("所有测试完成！")
    print("=" * 60)