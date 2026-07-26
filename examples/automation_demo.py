"""
自动化运行演示

运行3个周期，展示完整的：
认领→训练→评估→奖励→淘汰→导师加成 闭环
"""

import sys


def run_automation_demo():
    from xuni import AutomationRunner

    print("=" * 60)
    print("XUNI AUTOMATION DEMO")
    print("=" * 60)
    print("\nRunning 3 cycles...\n")

    runner = AutomationRunner()
    runner.run_cycles(n=3, verbose=True)

    print("\n" + runner.visualize())

    # 详细报告
    report = runner.get_report()
    print(f"\n--- Final Report ---")
    print(f"Cycles run: {report['cycles_run']}")
    print(f"Total models: {report['system']['total_models']}")
    print(f"Trained: {report['system']['total_trained']}")
    print(f"Total AI: {report['economy']['total_ai']}")
    print(f"Total energy: {report['economy']['total_energy']}")
    print(f"Mentors: {len(report['mentors'])}")

    print(f"\n--- Top 5 AI by Energy ---")
    for i, acc in enumerate(report["leaderboard"]):
        if acc:
            print(f"  {i+1}. {acc['owner']}: {acc['balance']} "
                  f"(earned={acc['total_earned']}, spent={acc['total_spent']}, lost={acc['total_lost']})")

    print(f"\n--- Top 5 Models by Score ---")
    for i, (mid, score) in enumerate(report["model_ranking"]):
        print(f"  {i+1}. {mid}: {score:.1f} pts")

    print("\n" + "=" * 60)
    print("AUTOMATION DEMO COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    run_automation_demo()
