"""
查看工厂生成的代码训练素材——质量如何？
然后把它丢进九宫骨架
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xuni.multiverse_resources import MultiverseResourceFactory
from xuni.training_forge import TrainingForge

print("=" * 78)
print("工厂生成的代码训练素材——质量审查")
print("=" * 78)

factory = MultiverseResourceFactory()

# 1. 基础生产：1000 条代码，最低 C 级
print("\n【1】基础生产：1000 条代码素材（最低 C 级）")
print("─" * 78)
result = factory.produce_training_data(count=1000, data_type="code", min_grade="C")
print(f"  生产总数: {result['total']}")
print(f"  平均质量分: {result['avg_quality']:.4f}")
print(f"  等级分布: {result['grade_distribution']}")

print("\n  抽样前 5 条:")
for i, text in enumerate(result['texts'][:5]):
    score = result['scores'][i]
    grade = result['grades'][i]
    grade_names = ["D","C","B","A","S","SS","SSS"]
    print(f"\n  ─── 样本 {i+1} | 质量={score:.4f} | 等级={grade_names[grade]} ───")
    print(text)

# 2. 能量锻造：100 条 S 级代码
print("\n\n【2】能量锻造：100 条代码（目标 S 级，能量=500）")
print("─" * 78)
result2 = factory.produce_training_data_energy(
    count=100, energy=500.0, data_type="code", target_grade="S"
)
print(f"  生产总数: {result2['total']}")
print(f"  平均质量分: {result2['avg_quality']:.4f}")
print(f"  等级分布: {result2['grade_distribution']}")
print(f"  注入能量: {result2.get('energy_used', '?')}")

print("\n  抽样前 3 条 S 级:")
for i, text in enumerate(result2['texts'][:3]):
    score = result2['scores'][i]
    grade = result2['grades'][i]
    grade_names = ["D","C","B","A","S","SS","SSS"]
    print(f"\n  ─── 样本 {i+1} | 质量={score:.4f} | 等级={grade_names[grade]} ───")
    print(text)

# 3. 淬炼训练素材：训练素材 + 质量点
print("\n\n【3】淬炼训练素材：1000 条（训练素材+质量点，S 级）")
print("─" * 78)
result3 = factory.produce_refined_training_data(count=1000, min_grade="S", quality_min_grade="A")
print(f"  生产总数: {result3.get('total', len(result3.get('texts',[])))}")
print(f"  平均质量分: {result3.get('avg_quality', '?')}")
print(f"  质量点使用: {result3.get('quality_points_used', '?')}")
print(f"  提升倍率: {result3.get('quality_boost', '?')}")

print("\n  抽样前 3 条淬炼代码:")
texts3 = result3.get('texts', [])
for i in range(min(3, len(texts3))):
    print(f"\n  ─── 淬炼样本 {i+1} ───")
    print(texts3[i] if len(texts3[i]) < 400 else texts3[i][:400] + "...")

print("\n" + "=" * 78)
print("代码质量评估总结")
print("=" * 78)
print("看得懂吗？请人工评审上面输出 😂")
