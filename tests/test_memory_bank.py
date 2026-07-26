"""
Tests for MemoryBank (from CEE extraction)
"""

import pytest
import time

from xuni.memory import (
    MemoryBank,
    ShortTermMemory,
    LongTermMemory,
    MemoryEntry,
    MemoryType,
    MemoryScope,
)


class TestShortTermMemory:
    def test_store_and_retrieve(self):
        stm = ShortTermMemory(capacity=3)
        e1 = MemoryEntry(memory_id="1", content="test1", importance=0.5)
        stm.store(e1)
        assert stm.size == 1
        retrieved = stm.retrieve("1")
        assert retrieved is not None
        assert retrieved.content == "test1"
        assert retrieved.access_count == 1

    def test_capacity_eviction(self):
        stm = ShortTermMemory(capacity=2)
        stm.store(MemoryEntry(memory_id="1", content="a"))
        stm.store(MemoryEntry(memory_id="2", content="b"))
        stm.store(MemoryEntry(memory_id="3", content="c"))
        assert stm.size == 2
        assert stm.retrieve("1") is None
        assert stm.retrieve("3") is not None

    def test_retrieve_by_tag(self):
        stm = ShortTermMemory()
        stm.store(MemoryEntry(memory_id="1", content="a", tags=["music"]))
        stm.store(MemoryEntry(memory_id="2", content="b", tags=["field"]))
        results = stm.retrieve_by_tag("music")
        assert len(results) == 1
        assert results[0].memory_id == "1"


class TestLongTermMemory:
    def test_store_and_search(self):
        ltm = LongTermMemory()
        ltm.store(MemoryEntry(memory_id="1", content="resonance pattern A", tags=["music", "pattern"], importance=0.8))
        ltm.store(MemoryEntry(memory_id="2", content="field config B", tags=["field"], importance=0.3))
        results = ltm.search_by_tag("music")
        assert len(results) == 1
        assert results[0].memory_id == "1"

    def test_search_by_content(self):
        ltm = LongTermMemory()
        ltm.store(MemoryEntry(memory_id="1", content="alpha wave"))
        ltm.store(MemoryEntry(memory_id="2", content="beta wave"))
        results = ltm.search_by_content("alpha")
        assert len(results) == 1

    def test_forget_below(self):
        ltm = LongTermMemory()
        ltm.store(MemoryEntry(memory_id="1", content="a", importance=0.05))
        ltm.store(MemoryEntry(memory_id="2", content="b", importance=0.9))
        n = ltm.forget_below(0.1)
        assert n == 1
        assert len(ltm._store) == 1

    def test_get_top_k(self):
        ltm = LongTermMemory()
        ltm.store(MemoryEntry(memory_id="1", content="a", importance=0.3))
        ltm.store(MemoryEntry(memory_id="2", content="b", importance=0.9))
        top = ltm.get_top_k(1)
        assert len(top) == 1
        assert top[0].memory_id == "2"


class TestMemoryBank:
    def test_memorize_and_recall(self):
        bank = MemoryBank()
        entry = bank.memorize("Test resonance", importance=0.7, tags=["test"])
        assert entry.memory_id is not None
        recalled = bank.recall(entry.memory_id)
        assert recalled is not None
        assert recalled.content == "Test resonance"

    def test_consolidation(self):
        bank = MemoryBank()
        e1 = bank.memorize("freq 440Hz", importance=0.4, tags=["freq"])
        # 模拟多次访问
        for _ in range(3):
            bank.recall(e1.memory_id)
        promoted = bank.consolidate()
        assert promoted >= 1
        assert e1.memory_id in bank.ltm._store

    def test_search(self):
        bank = MemoryBank()
        bank.memorize("pattern A", tags=["pattern"])
        bank.memorize("pattern B", tags=["pattern"])
        results = bank.search(tag="pattern")
        assert len(results) == 2

    def test_report(self):
        bank = MemoryBank()
        bank.memorize("x", importance=0.8)
        report = bank.report()
        assert "stm_size" in report
        assert "ltm_size" in report
        assert report["ltm_size"] >= 1
