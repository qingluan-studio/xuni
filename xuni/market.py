"""
XuniMarket —— 参数包交易市场

核心理念：
  AI 之间用能量交易参数包。
  参数包是模型的本质，交易参数包 = 交易模型能力。

市场机制：
  1. 挂卖：AI 把参数包挂到市场，定价（用能量）
  2. 购买：其他 AI 用能量购买参数包
  3. 拍卖：多个买家竞价，价高者得
  4. 价格由质量评分 + 供需决定

价格公式：
  base_price = quality * 0.5  （质量越高越贵）
  final_price = base_price * demand_multiplier
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List
from enum import Enum, auto

from .parameter import ParameterPack


class ListingStatus(Enum):
    """挂单状态"""
    ACTIVE = auto()      # 在售
    SOLD = auto()        # 已售
    CANCELLED = auto()   # 已取消
    EXPIRED = auto()     # 已过期


class AuctionStatus(Enum):
    """拍卖状态"""
    OPEN = auto()        # 进行中
    CLOSED = auto()      # 已结束
    CANCELLED = auto()   # 已取消


@dataclass
class Listing:
    """挂单"""
    listing_id: str
    seller: str                    # 卖家 AI
    pack: ParameterPack            # 参数包
    price: float                   # 价格（能量）
    status: ListingStatus = ListingStatus.ACTIVE
    created_at: float = 0.0
    sold_at: float = 0.0
    buyer: Optional[str] = None    # 买家


@dataclass
class Bid:
    """竞价"""
    bidder: str        # 竞价者
    amount: float      # 出价
    timestamp: float


@dataclass
class Auction:
    """拍卖"""
    auction_id: str
    seller: str
    pack: ParameterPack
    starting_price: float           # 起拍价
    current_highest_bid: float = 0.0
    highest_bidder: Optional[str] = None
    bids: List[Bid] = field(default_factory=list)
    status: AuctionStatus = AuctionStatus.OPEN
    created_at: float = 0.0
    closes_at: float = 0.0          # 结束时间


@dataclass
class TradeRecord:
    """交易记录"""
    trade_id: str
    seller: str
    buyer: str
    pack_id: str
    price: float
    timestamp: float
    trade_type: str  # "listing" or "auction"


class ParameterMarket:
    """
    参数包交易市场。
    
    AI 在这里买卖参数包，用能量交易。
    """

    def __init__(self, economy=None):
        self.economy = economy  # 关联能量经济学系统
        self.listings: Dict[str, Listing] = {}
        self.auctions: Dict[str, Auction] = {}
        self.trade_history: List[TradeRecord] = []
        self._listing_counter = 0
        self._auction_counter = 0

    def list_pack(self, seller: str, pack: ParameterPack, price: Optional[float] = None) -> Optional[str]:
        """
        挂卖参数包。
        
        如果不指定价格，根据质量自动定价。
        """
        if price is None:
            price = self._auto_price(pack)
        
        self._listing_counter += 1
        listing_id = f"lst-{self._listing_counter:04d}"
        
        listing = Listing(
            listing_id=listing_id,
            seller=seller,
            pack=pack,
            price=price,
            created_at=time.time(),
        )
        self.listings[listing_id] = listing
        return listing_id

    def buy(self, buyer: str, listing_id: str) -> Optional[TradeRecord]:
        """购买挂单"""
        listing = self.listings.get(listing_id)
        if listing is None or listing.status != ListingStatus.ACTIVE:
            return None
        
        # 检查能量
        if self.economy:
            buyer_balance = self.economy.get_balance(buyer)
            if buyer_balance < listing.price:
                return None  # 能量不足
            
            # 转账
            self.economy.redistribute_energy(buyer, listing.seller, listing.price)
        
        # 完成交易
        listing.status = ListingStatus.SOLD
        listing.buyer = buyer
        listing.sold_at = time.time()
        
        trade = TradeRecord(
            trade_id=f"trd-{len(self.trade_history)+1:04d}",
            seller=listing.seller,
            buyer=buyer,
            pack_id=listing.pack.pack_id,
            price=listing.price,
            timestamp=time.time(),
            trade_type="listing",
        )
        self.trade_history.append(trade)
        return trade

    def create_auction(
        self,
        seller: str,
        pack: ParameterPack,
        starting_price: Optional[float] = None,
        duration: float = 3600.0,
    ) -> Optional[str]:
        """创建拍卖"""
        if starting_price is None:
            starting_price = self._auto_price(pack) * 0.5  # 起拍价为定价的一半
        
        self._auction_counter += 1
        auction_id = f"auc-{self._auction_counter:04d}"
        
        auction = Auction(
            auction_id=auction_id,
            seller=seller,
            pack=pack,
            starting_price=starting_price,
            current_highest_bid=starting_price,
            created_at=time.time(),
            closes_at=time.time() + duration,
        )
        self.auctions[auction_id] = auction
        return auction_id

    def place_bid(self, bidder: str, auction_id: str, amount: float) -> bool:
        """竞价"""
        auction = self.auctions.get(auction_id)
        if auction is None or auction.status != AuctionStatus.OPEN:
            return False
        if time.time() > auction.closes_at:
            auction.status = AuctionStatus.CLOSED
            return False
        if amount <= auction.current_highest_bid:
            return False
        
        # 检查能量
        if self.economy:
            if self.economy.get_balance(bidder) < amount:
                return False
        
        bid = Bid(bidder=bidder, amount=amount, timestamp=time.time())
        auction.bids.append(bid)
        auction.current_highest_bid = amount
        auction.highest_bidder = bidder
        return True

    def close_auction(self, auction_id: str) -> Optional[TradeRecord]:
        """结束拍卖"""
        auction = self.auctions.get(auction_id)
        if auction is None or auction.status != AuctionStatus.OPEN:
            return None
        
        auction.status = AuctionStatus.CLOSED
        
        if auction.highest_bidder is None:
            return None  # 无人竞价
        
        # 转账
        if self.economy:
            self.economy.redistribute_energy(
                auction.highest_bidder, auction.seller, auction.current_highest_bid
            )
        
        trade = TradeRecord(
            trade_id=f"trd-{len(self.trade_history)+1:04d}",
            seller=auction.seller,
            buyer=auction.highest_bidder,
            pack_id=auction.pack.pack_id,
            price=auction.current_highest_bid,
            timestamp=time.time(),
            trade_type="auction",
        )
        self.trade_history.append(trade)
        return trade

    def cancel_listing(self, seller: str, listing_id: str) -> bool:
        """取消挂单"""
        listing = self.listings.get(listing_id)
        if listing is None or listing.seller != seller:
            return False
        if listing.status != ListingStatus.ACTIVE:
            return False
        listing.status = ListingStatus.CANCELLED
        return True

    def get_active_listings(self) -> List[Dict[str, Any]]:
        """获取在售挂单"""
        return [
            {
                "listing_id": l.listing_id,
                "seller": l.seller,
                "pack_id": l.pack.pack_id,
                "pack_source": l.pack.source,
                "quality": l.pack.quality,
                "price": l.price,
                "param_count": len(l.pack.params),
            }
            for l in self.listings.values()
            if l.status == ListingStatus.ACTIVE
        ]

    def get_open_auctions(self) -> List[Dict[str, Any]]:
        """获取进行中的拍卖"""
        now = time.time()
        return [
            {
                "auction_id": a.auction_id,
                "seller": a.seller,
                "pack_id": a.pack.pack_id,
                "quality": a.pack.quality,
                "starting_price": a.starting_price,
                "current_bid": a.current_highest_bid,
                "highest_bidder": a.highest_bidder,
                "bid_count": len(a.bids),
                "time_left": max(0, a.closes_at - now),
            }
            for a in self.auctions.values()
            if a.status == AuctionStatus.OPEN and now <= a.closes_at
        ]

    def _auto_price(self, pack: ParameterPack) -> float:
        """根据质量自动定价"""
        base = pack.quality * 0.5  # 质量100 → 50能量
        # 来源加成
        source_bonus = {"model": 1.3, "sampler": 1.0, "field": 0.8, "merged": 1.5}
        multiplier = source_bonus.get(pack.source, 1.0)
        return round(base * multiplier, 1)

    def statistics(self) -> Dict[str, Any]:
        """市场统计"""
        total_volume = sum(t.price for t in self.trade_history)
        return {
            "active_listings": sum(1 for l in self.listings.values() if l.status == ListingStatus.ACTIVE),
            "sold_listings": sum(1 for l in self.listings.values() if l.status == ListingStatus.SOLD),
            "open_auctions": sum(1 for a in self.auctions.values() if a.status == AuctionStatus.OPEN),
            "total_trades": len(self.trade_history),
            "total_volume": round(total_volume, 1),
            "avg_price": round(total_volume / len(self.trade_history), 1) if self.trade_history else 0,
        }
