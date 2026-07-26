"""
XuniEconomy —— 能量经济学系统

核心理念：
  AI 需要通过输出质量"赚"能量。
  能量不足的 AI 不能训练新模型。
  形成：训练→评估→赚能量→训练更多 的闭环。

经济模型：
  1. 每个 AI 有能量账户
  2. 训练模型消耗能量
  3. 模型被评估后，根据分数获得能量奖励
  4. 模型被调用时，产出能量（劳动）
  5. 导师模型有额外补贴
  6. 低分模型被淘汰，AI 损失投入的能量

能量分配：
  - 初始能量：每个 AI 100 点
  - 训练消耗：energy_requirement * 2
  - 评估奖励：score / 10（满分10点）
  - 调用产出：0.5 点/次
  - 导师补贴：2 点/评估周期
  - 淘汰损失：投入能量的 50% 不返还
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List

import numpy as np


@dataclass
class EnergyAccount:
    """AI 能量账户"""
    owner: str
    balance: float = 100.0             # 当前余额
    total_earned: float = 0.0          # 总收入
    total_spent: float = 0.0           # 总支出
    total_lost: float = 0.0            # 总损失（淘汰）
    models_owned: List[str] = field(default_factory=list)  # 拥有的模型
    models_retired: int = 0            # 被淘汰的模型数
    last_reward_time: float = 0.0      # 上次奖励时间


class EnergyEconomy:
    """
    能量经济学系统。
    
    管理 AI 能量账户，根据模型表现分配能量。
    """

    def __init__(
        self,
        initial_balance: float = 100.0,
        train_cost_multiplier: float = 2.0,     # 训练消耗 = energy_requirement * 此倍数
        eval_reward_divisor: float = 10.0,      # 评估奖励 = score / 此值
        call_reward: float = 0.5,               # 每次调用产出
        mentor_subsidy: float = 2.0,            # 导师补贴
        retirement_loss_rate: float = 0.5,      # 淘汰时损失投入的比例
    ):
        self.initial_balance = initial_balance
        self.train_cost_multiplier = train_cost_multiplier
        self.eval_reward_divisor = eval_reward_divisor
        self.call_reward = call_reward
        self.mentor_subsidy = mentor_subsidy
        self.retirement_loss_rate = retirement_loss_rate
        
        self.accounts: Dict[str, EnergyAccount] = {}
        self._training_investments: Dict[str, float] = {}  # model_id -> 投入能量

    def register_ai(self, owner: str) -> EnergyAccount:
        """注册 AI 账户"""
        if owner not in self.accounts:
            self.accounts[owner] = EnergyAccount(owner=owner, balance=self.initial_balance)
        return self.accounts[owner]

    def can_afford_training(self, owner: str, energy_requirement: float) -> bool:
        """检查 AI 是否能负担训练"""
        account = self.accounts.get(owner)
        if account is None:
            return False
        cost = energy_requirement * self.train_cost_multiplier
        return account.balance >= cost

    def charge_training(self, owner: str, model_id: str, energy_requirement: float) -> bool:
        """
        扣除训练费用。
        
        返回 False 表示能量不足。
        """
        account = self.accounts.get(owner)
        if account is None:
            return False
        
        cost = energy_requirement * self.train_cost_multiplier
        if account.balance < cost:
            return False
        
        account.balance -= cost
        account.total_spent += cost
        account.models_owned.append(model_id)
        self._training_investments[model_id] = cost
        return True

    def reward_evaluation(self, owner: str, model_id: str, score: float, is_mentor: bool = False):
        """
        根据评估分数发放能量奖励。
        """
        account = self.accounts.get(owner)
        if account is None:
            return
        
        # 评估奖励
        reward = score / self.eval_reward_divisor
        # 导师补贴
        if is_mentor:
            reward += self.mentor_subsidy
        
        account.balance += reward
        account.total_earned += reward
        account.last_reward_time = time.time()

    def reward_call(self, owner: str):
        """模型被调用时产出能量"""
        account = self.accounts.get(owner)
        if account is None:
            return
        account.balance += self.call_reward
        account.total_earned += self.call_reward

    def penalize_retirement(self, owner: str, model_id: str):
        """模型被淘汰，AI 损失投入的能量"""
        account = self.accounts.get(owner)
        if account is None:
            return
        
        investment = self._training_investments.get(model_id, 0)
        loss = investment * self.retirement_loss_rate
        account.balance -= loss
        account.total_lost += loss
        account.models_retired += 1
        
        if model_id in account.models_owned:
            account.models_owned.remove(model_id)
        if model_id in self._training_investments:
            del self._training_investments[model_id]

    def refund_training(self, owner: str, model_id: str):
        """退还训练投入（模型释放时）"""
        account = self.accounts.get(owner)
        if account is None:
            return
        
        investment = self._training_investments.get(model_id, 0)
        refund = investment * (1 - self.retirement_loss_rate)
        account.balance += refund
        account.total_earned += refund
        
        if model_id in account.models_owned:
            account.models_owned.remove(model_id)
        if model_id in self._training_investments:
            del self._training_investments[model_id]

    def get_balance(self, owner: str) -> float:
        """获取余额"""
        account = self.accounts.get(owner)
        return account.balance if account else 0.0

    def get_account_info(self, owner: str) -> Optional[Dict[str, Any]]:
        """获取账户信息"""
        account = self.accounts.get(owner)
        if account is None:
            return None
        return {
            "owner": account.owner,
            "balance": round(account.balance, 2),
            "total_earned": round(account.total_earned, 2),
            "total_spent": round(account.total_spent, 2),
            "total_lost": round(account.total_lost, 2),
            "net_profit": round(account.total_earned - account.total_spent - account.total_lost, 2),
            "models_owned": len(account.models_owned),
            "models_retired": account.models_retired,
            "model_ids": account.models_owned,
        }

    def get_leaderboard(self) -> List[Dict[str, Any]]:
        """获取能量排行榜"""
        ranked = sorted(self.accounts.values(), key=lambda a: a.balance, reverse=True)
        return [self.get_account_info(a.owner) for a in ranked]

    def statistics(self) -> Dict[str, Any]:
        """经济系统统计"""
        if not self.accounts:
            return {"total_ai": 0, "total_energy": 0}
        
        balances = [a.balance for a in self.accounts.values()]
        return {
            "total_ai": len(self.accounts),
            "total_energy": round(sum(balances), 2),
            "avg_balance": round(np.mean(balances), 2),
            "max_balance": round(max(balances), 2),
            "min_balance": round(min(balances), 2),
            "total_earned": round(sum(a.total_earned for a in self.accounts.values()), 2),
            "total_spent": round(sum(a.total_spent for a in self.accounts.values()), 2),
            "total_lost": round(sum(a.total_lost for a in self.accounts.values()), 2),
            "total_retired": sum(a.models_retired for a in self.accounts.values()),
        }

    def redistribute_energy(self, from_owner: str, to_owner: str, amount: float) -> bool:
        """AI 之间能量转账"""
        from_account = self.accounts.get(from_owner)
        to_account = self.accounts.get(to_owner)
        if from_account is None or to_account is None:
            return False
        if from_account.balance < amount:
            return False
        from_account.balance -= amount
        to_account.balance += amount
        return True
