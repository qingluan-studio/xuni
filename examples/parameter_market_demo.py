"""
参数系统 + 交易市场演示

围绕"参数"一词：
  采样点 → 抽取参数 → 参数包
  参数包 → 注入模型（强化）
  参数包 → 挂卖/拍卖（市场交易）
  参数包 → 导出/导入（JSON序列化）
"""

import numpy as np


def run_parameter_market_demo():
    from xuni import (
        XuniSampler, XuniField,
        ParameterExtractor, ParameterInjector, ParameterPack,
        ParameterMarket, EnergyEconomy,
        LayeredModelSystem, TrainingState,
    )

    print("=" * 60)
    print("PARAMETER & MARKET DEMO")
    print("=" * 60)

    # === 1. 采样点 → 参数包 ===
    print("\n[1/7] 采样点 → 参数包")
    sampler = XuniSampler(seed=42)
    samples = list(sampler.generate_stream(count=1000))
    pack_sampler = ParameterExtractor.from_samples(samples)
    print(f"  {pack_sampler.summary()}")
    print(f"  参数: {list(pack_sampler.params.keys())[:5]}...")

    # === 2. 模型 → 参数包（导出）===
    print("\n[2/7] 模型 → 参数包（导出）")
    system = LayeredModelSystem()
    system.setup_default_layers()
    system.auto_assign_from_pool()
    for layer in system.get_layers_ordered():
        for model in layer.models.values():
            if model.training_state == TrainingState.CLAIMED:
                model.start_training()
    system.train_until_complete(step_progress=0.5, max_steps=3)

    # 导出第一个已训练模型
    trained_model = None
    for layer in system.get_layers_ordered():
        for model in layer.models.values():
            if model.training_state == TrainingState.TRAINED:
                trained_model = model
                break
        if trained_model:
            break

    pack_model = ParameterExtractor.from_model(trained_model)
    print(f"  {pack_model.summary()}")
    print(f"  来源: {pack_model.origin_info}")

    # === 3. 参数包 → JSON 导出导入 ===
    print("\n[3/7] 参数包 JSON 导出/导入")
    json_str = pack_model.to_json()
    print(f"  导出 JSON: {len(json_str)} 字节")
    pack_imported = ParameterPack.from_json(json_str)
    print(f"  导入: {pack_imported.summary()}")
    print(f"  一致性: {pack_imported.pack_id == pack_model.pack_id}")

    # 保存到文件
    pack_model.save("/tmp/xuni_pack.json")
    pack_loaded = ParameterPack.load("/tmp/xuni_pack.json")
    print(f"  文件加载: {pack_loaded.summary()}")

    # === 4. 参数包 → 注入模型（强化）===
    print("\n[4/7] 参数包 → 注入模型")
    print(f"  注入前 energy_requirement: {trained_model.energy_requirement}")
    success = ParameterInjector.inject(trained_model, pack_sampler)
    print(f"  注入采样点参数包: {'成功' if success else '失败'}")
    print(f"  注入后 energy_requirement: {trained_model.energy_requirement}")

    # === 5. 合并参数包 ===
    print("\n[5/7] 合并参数包")
    merged = ParameterInjector.merge_packs([pack_sampler, pack_model])
    print(f"  {merged.summary()}")
    print(f"  来源包数: {merged.origin_info['merge_count']}")

    # === 6. 交易市场 ===
    print("\n[6/7] 交易市场")
    economy = EnergyEconomy()
    market = ParameterMarket(economy=economy)

    # 注册AI并给能量
    for name in ["Aria", "Bolt", "Coda"]:
        economy.register_ai(name)

    # Aria 挂卖模型参数包
    listing_id = market.list_pack("Aria", pack_model)
    listings = market.get_active_listings()
    print(f"  Aria 挂卖: {listings[0]['price']} 能量, 质量={listings[0]['quality']}")

    # Bolt 购买
    trade = market.buy("Bolt", listing_id)
    if trade:
        print(f"  Bolt 购买成功: {trade.price} 能量 → {trade.seller}")
    print(f"  Aria 余额: {economy.get_balance('Aria')}")
    print(f"  Bolt 余额: {economy.get_balance('Bolt')}")

    # === 7. 拍卖 ===
    print("\n[7/7] 拍卖")
    auction_id = market.create_auction("Coda", merged, duration=9999)
    market.place_bid("Aria", auction_id, 30.0)
    market.place_bid("Bolt", auction_id, 45.0)
    market.place_bid("Aria", auction_id, 60.0)
    auctions = market.get_open_auctions()
    print(f"  合并参数包拍卖: 当前最高出价 {auctions[0]['current_bid']} by {auctions[0]['highest_bidder']}")
    print(f"  竞价次数: {auctions[0]['bid_count']}")

    final_trade = market.close_auction(auction_id)
    if final_trade:
        print(f"  成交: {final_trade.buyer} 以 {final_trade.price} 能量购得")

    # 市场统计
    print(f"\n--- 市场统计 ---")
    stats = market.statistics()
    print(f"  总交易: {stats['total_trades']}")
    print(f"  总成交量: {stats['total_volume']} 能量")
    print(f"  平均价: {stats['avg_price']} 能量")

    print("\n" + "=" * 60)
    print("PARAMETER & MARKET DEMO COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    run_parameter_market_demo()
