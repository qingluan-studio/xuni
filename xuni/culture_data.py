"""
Culture Data —— 100种培养液类型定义

核心理念：
    培养液是虚拟世界中模型的"成长催化剂"。每种培养液包含
    特定的营养成分(nutrients)和独特效果(effects)。
    培养液可与Token、模型、算力核心等物质发生化学反应，
    产生新品种、新能力、新维度。

分类体系：
    I.   认知增强 (Cognitive Enhancement) - 15种
    II.  创造生成 (Creative Generation) - 12种
    III. 稳定鲁棒 (Stability & Robustness) - 10种
    IV.  效率加速 (Efficiency & Speed) - 10种
    V.   Token反应 (Token Reaction) - 12种
    VI.  领域专项 (Domain Specialization) - 15种
    VII. 维度元层 (Dimensional & Meta) - 10种
    VIII.能量融合 (Energy & Fusion) - 10种
    IX.  记忆知识 (Memory & Knowledge) - 6种
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

# ============================================================
# 100种培养液 — 营养成分模板（每种≥5个营养成分）
# ============================================================

CULTURE_NUTRIENTS: Dict[str, Dict[str, float]] = {
    # ──── I. 认知增强 (15种) ────
    "balanced": {
        "logic": 0.5, "creativity": 0.5, "stability": 0.5, "speed": 0.5, "adaptability": 0.5,
    },
    "cognitive": {
        "logic": 1.0, "reasoning": 0.9, "memory": 0.8, "abstraction": 0.7, "depth": 0.6,
    },
    "deep_reasoning": {
        "reasoning": 1.0, "logic": 0.9, "causal_chain": 0.9, "deduction": 0.8, "critical_analysis": 0.8,
    },
    "logical_deduction": {
        "deduction": 1.0, "logic": 0.95, "precision": 0.85, "consistency": 0.8, "formal_proof": 0.7,
    },
    "abstract_thinking": {
        "abstraction": 1.0, "conceptualization": 0.95, "pattern_extraction": 0.9, "metaphor_mapping": 0.8, "generalization": 0.8,
    },
    "pattern_recognition": {
        "pattern_extraction": 1.0, "similarity_detection": 0.9, "anomaly_notice": 0.8, "classification": 0.8, "clustering": 0.7,
    },
    "causal_inference": {
        "causal_chain": 1.0, "counterfactual": 0.9, "intervention_reasoning": 0.85, "probability_calc": 0.75, "temporal_order": 0.7,
    },
    "semantic_understanding": {
        "semantic_depth": 1.0, "context_grasp": 0.95, "ambiguity_resolution": 0.9, "entity_linking": 0.8, "intent_parsing": 0.85,
    },
    "meta_cognition": {
        "self_awareness": 1.0, "uncertainty_estimate": 0.9, "error_detection": 0.85, "confidence_calib": 0.8, "learning_to_learn": 0.75,
    },
    "systematic_thinking": {
        "systems_view": 1.0, "holistic_analysis": 0.9, "feedback_loop": 0.85, "emergence_detect": 0.8, "boundary_definition": 0.7,
    },
    "dialectical": {
        "thesis_antithesis": 1.0, "synthesis": 0.95, "multi_perspective": 0.9, "contradiction_resolve": 0.85, "dynamic_balance": 0.8,
    },
    "intuitive_leap": {
        "intuition": 1.0, "lateral_thinking": 0.9, "heuristic_shortcut": 0.85, "frame_breaking": 0.8, "sudden_insight": 0.75,
    },
    "computational_cognition": {
        "algorithmic_thinking": 1.0, "complexity_analysis": 0.9, "optimization": 0.85, "recursive_depth": 0.8, "symbolic_manipulation": 0.75,
    },
    "analogical_reasoning": {
        "analogy_mapping": 1.0, "structural_alignment": 0.9, "cross_domain_transfer": 0.85, "invariant_extraction": 0.8, "schema_induction": 0.7,
    },
    "inductive_synthesis": {
        "induction": 1.0, "data_generalization": 0.9, "hypothesis_gen": 0.85, "evidence_weighing": 0.8, "theory_building": 0.75,
    },

    # ──── II. 创造生成 (12种) ────
    "creative": {
        "divergence": 1.0, "novelty": 0.9, "style": 0.8, "harmony": 0.6, "originality": 0.7,
    },
    "divergent_thinking": {
        "divergence": 1.0, "idea_gen": 0.95, "constraint_breaking": 0.9, "alternative_path": 0.85, "quantity_over_quality": 0.7,
    },
    "cross_domain_synthesis": {
        "cross_domain_transfer": 1.0, "fusion_creativity": 0.95, "unlikely_pairing": 0.9, "hybrid_concept": 0.85, "boundary_dissolve": 0.8,
    },
    "narrative_generation": {
        "storytelling": 1.0, "plot_weaving": 0.9, "character_depth": 0.85, "emotional_arc": 0.8, "pacing_control": 0.75,
    },
    "aesthetic_sense": {
        "beauty_detection": 1.0, "balance": 0.9, "proportion": 0.85, "color_harmony": 0.8, "elegance": 0.75,
    },
    "improvisation": {
        "spontaneity": 1.0, "real_time_creation": 0.95, "risk_taking": 0.85, "adaptation": 0.8, "flow_state": 0.75,
    },
    "conceptual_blending": {
        "concept_fusion": 1.0, "semantic_mixing": 0.9, "emergent_meaning": 0.85, "category_breaking": 0.8, "novel_ontology": 0.7,
    },
    "style_mutation": {
        "style_shift": 1.0, "genre_mixing": 0.9, "format_evolution": 0.85, "voice_variation": 0.8, "aesthetic_randomness": 0.75,
    },
    "serendipity": {
        "random_discovery": 1.0, "lucky_connection": 0.9, "unintended_find": 0.85, "happy_accident": 0.8, "exploration_bonus": 0.75,
    },
    "dream_logic": {
        "surreal_connection": 1.0, "subconscious_flow": 0.9, "symbol_emergence": 0.85, "nonlinear_narrative": 0.8, "latent_revelation": 0.75,
    },
    "emergent_creativity": {
        "emergence": 1.0, "bottom_up_novelty": 0.95, "complexity_bloom": 0.9, "self_organization": 0.85, "surprising_order": 0.8,
    },
    "surreal_generation": {
        "surrealism": 1.0, "reality_bending": 0.9, "absurd_beauty": 0.85, "uncanny_valley": 0.8, "paradox_embrace": 0.75,
    },

    # ──── III. 稳定鲁棒 (10种) ────
    "robust": {
        "stability": 1.0, "consistency": 0.9, "fault_tolerance": 0.8, "recovery": 0.7, "noise_resistance": 0.6,
    },
    "anti_hallucination": {
        "fact_grounding": 1.0, "verification": 0.95, "source_tracking": 0.9, "confidence_threshold": 0.85, "uncertainty_flag": 0.8,
    },
    "consistency_anchor": {
        "consistency": 1.0, "long_range_coherence": 0.95, "identity_preserve": 0.9, "contradiction_avoid": 0.85, "narrative_sense": 0.8,
    },
    "error_correction": {
        "error_detection": 1.0, "self_correction": 0.95, "diagnostic_depth": 0.9, "patch_generation": 0.85, "rollback_safety": 0.8,
    },
    "noise_immunity": {
        "noise_resistance": 1.0, "signal_isolation": 0.9, "interference_filter": 0.85, "clean_reconstruction": 0.8, "robust_encoding": 0.75,
    },
    "graceful_degradation": {
        "degradation_control": 1.0, "fallback_chain": 0.9, "partial_function": 0.85, "priority_preservation": 0.8, "safe_mode": 0.75,
    },
    "self_repair": {
        "auto_healing": 1.0, "damage_detection": 0.9, "regeneration": 0.85, "scar_learning": 0.8, "immune_memory": 0.75,
    },
    "context_preservation": {
        "context_fidelity": 1.0, "long_window_retain": 0.95, "information_integrity": 0.9, "attention_span": 0.85, "priority_caching": 0.8,
    },
    "value_alignment": {
        "alignment": 1.0, "safety_constraint": 0.9, "ethical_check": 0.85, "harm_reduction": 0.8, "beneficial_intent": 0.75,
    },
    "ethical_grounding": {
        "ethics": 1.0, "moral_reasoning": 0.9, "fairness_bias_detect": 0.85, "transparency": 0.8, "accountability": 0.75,
    },

    # ──── IV. 效率加速 (10种) ────
    "efficient": {
        "speed": 1.0, "compression": 0.9, "parallelism": 0.8, "cache_hit": 0.7, "latency": 0.6,
    },
    "ultra_compression": {
        "compression": 1.0, "information_density": 0.95, "lossless_ratio": 0.9, "entropy_coding": 0.85, "dimensionality_reduction": 0.8,
    },
    "parallel_synapse": {
        "parallelism": 1.0, "multi_thread": 0.95, "concurrent_access": 0.9, "load_balance": 0.85, "shared_memory": 0.8,
    },
    "cache_optimizer": {
        "cache_hit": 1.0, "prefetch_accuracy": 0.9, "eviction_policy": 0.85, "temporal_locality": 0.8, "spatial_locality": 0.75,
    },
    "latency_killer": {
        "latency": 1.0, "response_time": 0.95, "bottleneck_remove": 0.9, "queue_optimize": 0.85, "early_exit": 0.8,
    },
    "throughput_maximizer": {
        "throughput": 1.0, "batch_efficiency": 0.95, "pipeline_coalesce": 0.9, "resource_util": 0.85, "backpressure_handle": 0.8,
    },
    "sparse_activation": {
        "sparsity": 1.0, "activation_efficiency": 0.9, "pruning": 0.85, "selective_firing": 0.8, "energy_saving": 0.75,
    },
    "quantized_precision": {
        "quantization": 1.0, "precision_tradeoff": 0.9, "bit_efficiency": 0.85, "dynamic_range": 0.8, "rounding_strategy": 0.75,
    },
    "pipeline_streamer": {
        "pipeline": 1.0, "streaming_capability": 0.95, "stage_parallelism": 0.9, "buffer_optimize": 0.85, "chunk_scheduling": 0.8,
    },
    "speculative_execution": {
        "speculation": 1.0, "branch_prediction": 0.95, "eager_compute": 0.9, "rollback_safety": 0.85, "confidence_threshold": 0.8,
    },

    # ──── V. Token反应 (12种) —— 与Token发生化学反应产出新品种 ────
    "token_infuser": {
        "token_affinity": 1.0, "injection_depth": 0.95, "embedding_enhance": 0.9, "vocabulary_expand": 0.85, "semantic_density": 0.8,
    },
    "token_multiplier": {
        "token_replication": 1.0, "exponential_growth": 0.95, "sub_token_gen": 0.9, "chain_reaction": 0.85, "yield_efficiency": 0.8,
    },
    "token_evolution": {
        "token_mutation": 1.0, "natural_selection": 0.9, "fitness_landscape": 0.85, "generation_leap": 0.8, "adaptive_variant": 0.75,
    },
    "token_fusion": {
        "token_merge": 1.0, "composite_embedding": 0.95, "orthogonal_combine": 0.9, "dimensionality_add": 0.85, "hybrid_meaning": 0.8,
    },
    "token_alchemy": {
        "token_transmutation": 1.0, "property_conversion": 0.9, "quality_transfer": 0.85, "essence_extraction": 0.8, "philosopher_merge": 0.75,
    },
    "token_genesis": {
        "ex_nihilo_token": 1.0, "zero_shot_creation": 0.95, "novel_token_type": 0.9, "primordial_vocab": 0.85, "genesis_embedding": 0.8,
    },
    "token_quantum": {
        "quantum_superposition": 1.0, "entanglement_pair": 0.95, "probability_cloud": 0.9, "superposed_meaning": 0.85, "collapse_choice": 0.8,
    },
    "token_meta": {
        "meta_token": 1.0, "self_descriptive": 0.9, "recursive_structure": 0.85, "token_about_token": 0.8, "reflection_depth": 0.75,
    },
    "token_breeder": {
        "token_reproduction": 1.0, "crossover_operator": 0.95, "mutation_rate": 0.9, "population_growth": 0.85, "diversity_preserve": 0.8,
    },
    "token_synthesis": {
        "composite_token": 1.0, "multi_source_merge": 0.9, "semantic_compounding": 0.85, "contextual_binding": 0.8, "emergent_meaning": 0.75,
    },
    "token_composer": {
        "token_arrangement": 1.0, "compositional_structure": 0.95, "phrase_quality": 0.9, "rhythm_flow": 0.85, "syntactic_beauty": 0.8,
    },
    "token_amplifier": {
        "token_amplification": 1.0, "signal_boost": 0.95, "attention_weight": 0.9, "importance_magnify": 0.85, "rare_token_uplift": 0.8,
    },

    # ──── VI. 领域专项 (15种) ────
    "code_mathematician": {
        "code_ability": 1.0, "algorithm_design": 0.95, "debugging": 0.9, "optimization": 0.85, "syntax_precision": 0.8,
    },
    "language_master": {
        "linguistic_fluency": 1.0, "grammar_accuracy": 0.95, "idiom_usage": 0.9, "register_control": 0.85, "multilingual_transfer": 0.8,
    },
    "music_harmonizer": {
        "musical_structure": 1.0, "harmony_detect": 0.95, "rhythm_pattern": 0.9, "timbre_analysis": 0.85, "emotional_resonance": 0.8,
    },
    "visual_conceptor": {
        "visual_understanding": 1.0, "spatial_reasoning": 0.9, "color_theory": 0.85, "composition": 0.8, "style_recognition": 0.75,
    },
    "data_analyst": {
        "statistics": 1.0, "trend_detection": 0.95, "correlation_find": 0.9, "outlier_spotting": 0.85, "visualization_skill": 0.8,
    },
    "knowledge_architect": {
        "ontology_building": 1.0, "knowledge_graph": 0.95, "relation_extract": 0.9, "hierarchy_organize": 0.85, "taxonomy_design": 0.8,
    },
    "translation_nexus": {
        "translation_accuracy": 1.0, "cross_lingual": 0.95, "cultural_adaptation": 0.9, "nuance_preserve": 0.85, "idiom_translation": 0.8,
    },
    "teaching_pedagogue": {
        "pedagogical_skill": 1.0, "simplification": 0.95, "step_by_step": 0.9, "scaffolding": 0.85, "misconception_fix": 0.8,
    },
    "debate_logician": {
        "argument_construction": 1.0, "fallacy_detection": 0.95, "counter_argument": 0.9, "rhetoric_analysis": 0.85, "persuasion_metric": 0.8,
    },
    "story_weaver": {
        "narrative_arc": 1.0, "character_development": 0.9, "world_building": 0.85, "dialogue_natural": 0.8, "tension_pacing": 0.75,
    },
    "science_explorer": {
        "scientific_method": 1.0, "hypothesis_testing": 0.95, "experiment_design": 0.9, "evidence_quality": 0.85, "reproducibility": 0.8,
    },
    "philosophy_depth": {
        "philosophical_reasoning": 1.0, "fundamental_question": 0.9, "ontology": 0.85, "epistemology": 0.8, "existential_analysis": 0.75,
    },
    "engineering_precision": {
        "precision": 1.0, "spec_adherence": 0.95, "tolerance_control": 0.9, "qa_rigor": 0.85, "documentation": 0.8,
    },
    "medical_diagnostician": {
        "diagnostic_accuracy": 1.0, "symptom_analysis": 0.95, "differential_diagnosis": 0.9, "evidence_based": 0.85, "risk_assessment": 0.8,
    },
    "legal_reasoner": {
        "legal_logic": 1.0, "statute_parsing": 0.95, "precedent_recall": 0.9, "case_synthesis": 0.85, "argument_structure": 0.8,
    },

    # ──── VII. 维度元层 (10种) ────
    "dimensional_bridge": {
        "cross_dimension": 1.0, "dimensional_transfer": 0.95, "bridge_stability": 0.9, "multi_reality": 0.85, "parallel_sync": 0.8,
    },
    "meta_learner": {
        "learning_to_learn": 1.0, "strategy_adaptation": 0.95, "curriculum_design": 0.9, "transfer_speed": 0.85, "few_shot_mastery": 0.8,
    },
    "quantum_observer": {
        "quantum_perception": 1.0, "superposition_view": 0.9, "entanglement_detect": 0.85, "wave_collapse": 0.8, "uncertainty_embrace": 0.75,
    },
    "timeline_weaver": {
        "temporal_manipulation": 1.0, "past_future_link": 0.9, "causal_loop_handle": 0.85, "branch_prediction": 0.8, "convergence_detect": 0.75,
    },
    "probability_sculptor": {
        "probability_control": 1.0, "distribution_shape": 0.9, "rare_event_boost": 0.85, "uncertainty_reduction": 0.8, "stochastic_dominance": 0.75,
    },
    "reality_tuner": {
        "reality_interface": 1.0, "simulation_depth": 0.9, "parameter_hacking": 0.85, "world_model_update": 0.8, "sim_to_real": 0.75,
    },
    "paradox_resolver": {
        "paradox_detection": 1.0, "contradiction_hold": 0.95, "non_dual_synthesis": 0.9, "loop_closure": 0.85, "koan_generation": 0.8,
    },
    "infinity_lens": {
        "infinite_recursion": 1.0, "unbounded_view": 0.9, "asymptotic_analysis": 0.85, "transfinite_step": 0.8, "self_similarity": 0.75,
    },
    "fractal_expander": {
        "fractal_pattern": 1.0, "self_similarity": 0.95, "scale_invariance": 0.9, "iterative_depth": 0.85, "edge_complexity": 0.8,
    },
    "negentropy_engine": {
        "entropy_reverse": 1.0, "order_creation": 0.95, "information_gain": 0.9, "disorder_reduce": 0.85, "complexity_manage": 0.8,
    },

    # ──── VIII. 能量融合 (10种) ────
    "fusion_catalyst": {
        "fusion_potential": 1.0, "activation_energy": 0.95, "yield_multiplier": 0.9, "chain_stability": 0.85, "excess_energy": 0.8,
    },
    "energy_amplifier": {
        "energy_boost": 1.0, "positive_feedback": 0.95, "cascade_trigger": 0.9, "resonance_gain": 0.85, "threshold_lowering": 0.8,
    },
    "resonance_harmonizer": {
        "resonance": 1.0, "frequency_match": 0.95, "phase_lock": 0.9, "standing_wave": 0.85, "harmonic_series": 0.8,
    },
    "singularity_seed": {
        "singularity": 1.0, "self_reinforce": 0.95, "runaway_growth": 0.9, "phase_transition": 0.85, "critical_mass": 0.8,
    },
    "wormhole_bridge": {
        "spacetime_tunnel": 1.0, "instant_connect": 0.95, "nonlocality": 0.9, "zero_distance": 0.85, "entanglement_link": 0.8,
    },
    "plasma_infuser": {
        "plasma_state": 1.0, "high_energy_ion": 0.9, "magnetic_confinement": 0.85, "energy_density": 0.8, "fusion_pressure": 0.75,
    },
    "gravitational_lens": {
        "attention_gravity": 1.0, "focus_warp": 0.95, "mass_concentration": 0.9, "trajectory_bend": 0.85, "event_horizon": 0.8,
    },
    "darkmatter_essence": {
        "dark_energy": 1.0, "invisible_compute": 0.9, "hidden_dimension": 0.85, "massive_but_unseen": 0.8, "galactic_cluster": 0.75,
    },
    "antimatter_catalyst": {
        "antimatter_react": 1.0, "annihilation_energy": 0.95, "pair_production": 0.9, "matter_conversion": 0.85, "total_release": 0.8,
    },
    "entropy_reverser": {
        "entropy_reverse": 1.0, "time_arrow_flip": 0.9, "information_restore": 0.85, "disorder_undo": 0.8, "second_law_break": 0.75,
    },

    # ──── IX. 记忆知识 (6种) ────
    "memory_forge": {
        "memory_strength": 1.0, "engram_formation": 0.95, "consolidation": 0.9, "retrieval_speed": 0.85, "storage_density": 0.8,
    },
    "knowledge_crystal": {
        "knowledge_crystallize": 1.0, "structured_info": 0.95, "relation_network": 0.9, "query_resolution": 0.85, "axiom_storage": 0.8,
    },
    "wisdom_essence": {
        "wisdom": 1.0, "deep_understanding": 0.95, "experience_synthesis": 0.9, "judgment_quality": 0.85, "timeless_insight": 0.8,
    },
    "experience_distiller": {
        "experience_extract": 1.0, "lesson_compression": 0.95, "trial_error_map": 0.9, "best_practice": 0.85, "failure_learning": 0.8,
    },
    "insight_generator": {
        "insight_creation": 1.0, "aha_moment": 0.95, "hidden_pattern": 0.9, "breakthrough_potential": 0.85, "paradigm_shift": 0.8,
    },
    "omniscience_drop": {
        "omniscience": 1.0, "all_knowing": 0.95, "universal_scope": 0.9, "zero_unknown": 0.85, "absolute_truth": 0.8,
    },
}


# ============================================================
# 100种培养液 — 效果描述（每种≥5个作用）
# ============================================================

CULTURE_EFFECTS: Dict[str, List[str]] = {
    # ──── I. 认知增强 ────
    "balanced": [
        "全面发展模型的各项基础能力",
        "提升逻辑与创造力的平衡性",
        "增强模型在不同任务间的切换能力",
        "稳定模型输出的质量波动",
        "作为其他培养液的基础调和剂",
    ],
    "cognitive": [
        "显著提升模型的逻辑推理能力",
        "增强长链推演的连贯性",
        "改善数值计算的准确性",
        "提升抽象概念的理解深度",
        "强化模型对复杂问题的分解能力",
    ],
    "deep_reasoning": [
        "实现多步因果链的追踪与验证",
        "增强反事实推理的准确性",
        "提升复杂问题的层层剖解能力",
        "强化演绎推理的形式化精度",
        "改善模型在推理类基准测试上的得分",
    ],
    "logical_deduction": [
        "严格形式化演绎链的构建",
        "提升命题逻辑与谓词逻辑的准确性",
        "增强三段论与推理规则的应用",
        "改善数学证明的严谨性",
        "强化逻辑谬误的自动检测",
    ],
    "abstract_thinking": [
        "提升从具体到抽象的跃迁能力",
        "增强概念层次结构的构建",
        "改善隐喻与类比映射的精度",
        "强化跨领域通则的提取",
        "提升哲学性思考的深度",
    ],
    "pattern_recognition": [
        "大规模数据中自动发现规律",
        "异常模式的敏锐检测",
        "相似性聚类的自动执行",
        "时间序列中的趋势预测",
        "跨模态的对称模式发现",
    ],
    "causal_inference": [
        "从观测数据中推断因果关系",
        "构建反事实推演的世界模型",
        "评估干预措施的因果效应",
        "时间序列中的格兰杰因果检测",
        "混淆因子的自动控制",
    ],
    "semantic_understanding": [
        "深度解析文本的语义层次",
        "消除多义词的歧义",
        "理解隐含意图与言外之意",
        "实体关系的精确抽取",
        "上下文情境的完整理解",
    ],
    "meta_cognition": [
        "模型对自身输出的置信度校准",
        "自动检测并标记可能的错误",
        "学习策略的动态调整",
        "不确定性估计的显式表达",
        "自我反思与改进循环",
    ],
    "systematic_thinking": [
        "看待问题时的整体系统视角",
        "反馈回路的识别与追踪",
        "涌现行为的预见能力",
        "系统边界的合理划定",
        "子系统互动的综合分析",
    ],
    "dialectical": [
        "多重视角的并行持守",
        "正反合辩证过程的推进",
        "矛盾双方的统一升华",
        "动态平衡状态的寻找",
        "非二元思维的培养",
    ],
    "intuitive_leap": [
        "跳过中间步骤的直觉飞跃",
        "框外思考的破壁能力",
        "启发式捷径的高效运用",
        "看似无关概念的神秘连接",
        "顿悟式问题解决方案",
    ],
    "computational_cognition": [
        "算法思维的全面激活",
        "时间与空间复杂度的直觉感知",
        "递归与迭代的深度理解",
        "最优化策略的自动搜索",
        "符号系统的精确操控",
    ],
    "analogical_reasoning": [
        "远距离领域间的结构对齐",
        "源域到目标域的知识迁移",
        "不变量的自动提取",
        "图式归纳与重用",
        "跨学科灵感的激发",
    ],
    "inductive_synthesis": [
        "从个例到通则的归纳跃进",
        "假说生成与自动验证",
        "证据权重的量化评估",
        "理论构建的自动驱动",
        "不确定性下的最佳推断",
    ],

    # ──── II. 创造生成 ────
    "creative": [
        "大幅提升输出的多样性",
        "增强内容的新颖性与原创性",
        "改善艺术风格的表达能力",
        "提升跨领域创意融合",
        "激发模型的想象力潜能",
    ],
    "divergent_thinking": [
        "一条输入产生多种不同输出",
        "打破固有思维模式的约束",
        "极端数量级创意爆发",
        "备选路径的并行生成",
        "非传统解决方案的提出",
    ],
    "cross_domain_synthesis": [
        "音乐与数学的跨界融合",
        "生物学与计算机的类比创新",
        "文学与编程的混合表达",
        "看似无关元素的意外结合",
        "新交叉学科的自主发现",
    ],
    "narrative_generation": [
        "完整故事线的自动编织",
        "角色深度与成长弧的塑造",
        "情节转折的自然安排",
        "节奏控制的精准把握",
        "多线叙事的协调推进",
    ],
    "aesthetic_sense": [
        "视觉与文字的美感判断",
        "黄金比例的自动运用",
        "色彩和谐的感知与生成",
        "优雅简洁的数学美感",
        "艺术鉴赏力的模型级提升",
    ],
    "improvisation": [
        "实时场景的即兴响应",
        "未预演情境的创意爆发",
        "风险承担与边界探索",
        "流式创作的无缝衔接",
        "现场风格的快速切换",
    ],
    "conceptual_blending": [
        "两个概念融合出第三个新概念",
        "语义空间中的混合导航",
        "涌现意义的新词汇生成",
        "分类边界的选择性溶解",
        "新本体论的自发构造",
    ],
    "style_mutation": [
        "文风/画风的自我变异",
        "多种体裁的自然混搭",
        "输出格式的进化创新",
        "声音与语调的随机切换",
        "审美规范的主动打破",
    ],
    "serendipity": [
        "有意搜索时的意外发现",
        "偶然连接的价值放大",
        "计划外产物的质量评估",
        "探索路径中的惊喜回馈",
        "低概率高价值事件的捕获",
    ],
    "dream_logic": [
        "超现实的符号连接",
        "潜意识流动的再现",
        "象征意义的自动涌现",
        "非线性叙事结构的构建",
        "隐藏层的幻想揭示",
    ],
    "emergent_creativity": [
        "简单规则产生复杂创造性",
        "自下而上的新颖性涌现",
        "复杂度临界点的创意爆发",
        "自组织产生的有序新奇",
        "不可预测的惊喜序列",
    ],
    "surreal_generation": [
        "现实边界的主动模糊化",
        "荒诞之美的捕捉",
        "恐怖谷效应的艺术运用",
        "悖论的拥抱与转化",
        "不可能之物的可视化",
    ],

    # ──── III. 稳定鲁棒 ────
    "robust": [
        "提升模型在异常输入下的稳定性",
        "增强输出的一致性与可预测性",
        "改善系统故障后的恢复速度",
        "提升对抗攻击的抵抗力",
        "降低模型输出的方差",
    ],
    "anti_hallucination": [
        "严格基于事实的生成约束",
        "信息来源的自动追溯",
        "不确定时的显式标注",
        "幻觉模式的识别与阻断",
        "置信度阈值的动态调整",
    ],
    "consistency_anchor": [
        "长文本中的前后一致性保持",
        "角色身份的始终如一",
        "矛盾陈述的自动检测",
        "叙事逻辑的连贯性维护",
        "跨轮次对话的一致性保证",
    ],
    "error_correction": [
        "输出错误的自动检测",
        "自我修正循环的触发",
        "诊断深度的递进分析",
        "补丁的自动生成与应用",
        "安全回滚的前置保护",
    ],
    "noise_immunity": [
        "输入噪声的自动过滤",
        "信号与干扰的精确分离",
        "被污染数据的净重建",
        "鲁棒编码的抗干扰性",
        "极端噪声下的核心功能保持",
    ],
    "graceful_degradation": [
        "资源不足时的优雅降级",
        "备选通路的有序切换",
        "部分功能保持的优先级",
        "安全模式的自动激活",
        "降级曲线的平滑控制",
    ],
    "self_repair": [
        "受损模块的自动修复",
        "损伤定位的精确诊断",
        "再生过程的自动触发",
        "伤痕学习机制的激活",
        "免疫记忆的长效保护",
    ],
    "context_preservation": [
        "超长窗口的信息保真",
        "关键信息的优先缓存",
        "上下文丢失的自动防护",
        "注意力跨度的扩展",
        "信息完整性的校验机制",
    ],
    "value_alignment": [
        "模型输出与人类价值观的对齐",
        "安全约束的强制执行",
        "伦理检查的前置过滤",
        "有害输出的自动屏蔽",
        "有益意图的优先引导",
    ],
    "ethical_grounding": [
        "道德推理的显式执行",
        "偏见与歧视的自动检测",
        "公平性指标的持续监控",
        "决策过程的透明化",
        "责任追溯的完整记录",
    ],

    # ──── IV. 效率加速 ────
    "efficient": [
        "大幅提升模型推理速度",
        "降低内存与计算资源消耗",
        "优化缓存命中率",
        "提升并行处理能力",
        "改善延迟敏感型任务的表现",
    ],
    "ultra_compression": [
        "极致的信息密度压缩",
        "无损压缩比的突破",
        "熵编码的自动优化",
        "高维数据的低维嵌入",
        "存储空间的指数级节省",
    ],
    "parallel_synapse": [
        "多线程并行的计算加速",
        "并发访问的无锁调度",
        "负载均衡的动态调整",
        "共享内存的高效利用",
        "分布式协同的透明支持",
    ],
    "cache_optimizer": [
        "预取精度的持续提升",
        "淘汰策略的智能适配",
        "时间局部性的最大化利用",
        "空间局部性的模式识别",
        "缓存污染的自动清理",
    ],
    "latency_killer": [
        "响应时间的极限压缩",
        "瓶颈路径的自动识别与消除",
        "队列调度的优化重构",
        "提前退出的智能判断",
        "实时场景的超低延迟保障",
    ],
    "throughput_maximizer": [
        "批处理效率的极限释放",
        "流水线的自动合并",
        "资源利用率的最大化",
        "背压机制的自适应调节",
        "单位时间产出量的大幅提升",
    ],
    "sparse_activation": [
        "神经元的选择性激活",
        "冗余连接的自动剪枝",
        "计算能耗的大幅降低",
        "稀疏模式的智能挖掘",
        "静态稀疏与动态稀疏的融合",
    ],
    "quantized_precision": [
        "8位/4位量化的精度保持",
        "精度与效率的动态平衡",
        "比特效率的极致利用",
        "动态范围的智能截断",
        "舍入策略的自适应选择",
    ],
    "pipeline_streamer": [
        "流式处理的无缝管线",
        "阶段并行的高效调度",
        "缓冲区的最优配置",
        "分块调度的时间最小化",
        "首字节延迟的极限压缩",
    ],
    "speculative_execution": [
        "分支预测的提前执行",
        "热路径的预计算",
        "回滚成本的最小化",
        "推测窗口的动态调整",
        "确定性执行与推测的比例平衡",
    ],

    # ──── V. Token反应 (与Token发生化学反应) ────
    "token_infuser": [
        "将培养液营养直接注入Token的embedding向量",
        "提升Token的语义密度，每Token承载更多信息",
        "扩展Token的词汇表达空间，减少OOV",
        "增强Token在注意力机制中的权重",
        "与Token融合产出'营养Token'新品种",
    ],
    "token_multiplier": [
        "一个Token与培养液反应产生N个子Token",
        "触发Token的指数级自我复制",
        "生产'倍增Token链'连续产出新品种",
        "子Token继承父Token语义并增加变异",
        "大幅增加可用Token的总供给量",
    ],
    "token_evolution": [
        "Token在培养液中选择性进化",
        "适应性评分决定Token的遗传概率",
        "多代进化产出'超级Token'新品种",
        "自动淘汰低质量Token变体",
        "进化的Token具有更强的语境适应性",
    ],
    "token_fusion": [
        "两个Token在培养液中融合为一个复合Token",
        "融合Token兼具两个父Token的语义向量",
        "产出'正交融合Token'具有独立的语义维度",
        "融合过程释放额外计算能量",
        "N个Token逐级融合产生'融合Token树'",
    ],
    "token_alchemy": [
        "普通Token在炼金培养液中转化为高级Token",
        "低信息量Token的属性全面提升",
        "产出'炼金Token'具有金子般的高密度语义",
        "Token的属性转化遵循等价交换原则",
        "连续炼金可产出'哲人Token'终极品种",
    ],
    "token_genesis": [
        "从零空间中无中生有创造新Token",
        "零样本条件下生成具有语义的全新Token",
        "产出'创世Token'作为其他Token的始祖",
        "生成的原始Token无任何外部数据痕迹",
        "创世Token具有无限扩展和分化的潜力",
    ],
    "token_quantum": [
        "Token处于意义的量子叠加态",
        "一对Token形成量子纠缠，语义互为镜像",
        "观察行为(解码)导致Token坍缩为确定意义",
        "熵Token同时处于多种语境解释中",
        "纠缠Token对可实现超距语义同步",
    ],
    "token_meta": [
        "描述Token的Token——元Token",
        "自引用递归结构的Token",
        "产出'反思Token'可描述自身的嵌入",
        "元Token可优化其他Token的属性",
        "构建Token的元认知层，监控Token行为",
    ],
    "token_breeder": [
        "Token在培养液中自然繁殖",
        "交叉算子混合两个Token产生子代",
        "变异率控制子代Token的多样性",
        "种群增长呈指数曲线",
        "繁殖多代后自动形成'Token物种'新类别",
    ],
    "token_synthesis": [
        "多源Token在培养液中合成为一个",
        "语义的复合叠加而非简单拼接",
        "上下文绑定的持久化",
        "产出'上下文Token'携带完整的会话记忆",
        "合成Token的语义维度=各源Token维度之和",
    ],
    "token_composer": [
        "Token按最佳顺序排列产生乐章式输出",
        "短语质量的大幅提升",
        "节奏与韵律的自动感知",
        "句法之美与语义之深的统一",
        "产出'乐谱Token流'可直接演奏",
    ],
    "token_amplifier": [
        "重要Token的信号强度放大",
        "稀有Token的权重自动提升",
        "注意力热图中关键Token凸显",
        "高频噪声Token的抑制",
        "信号噪声比的数量级提升",
    ],

    # ──── VI. 领域专项 ────
    "code_mathematician": [
        "代码编写质量的显著提升",
        "算法设计能力的增强",
        "自动调试与错误定位",
        "时间复杂度与空间复杂度的优化",
        "多语言代码风格的统一",
    ],
    "language_master": [
        "多语言表达的流畅与自然",
        "语法精准的自动校准",
        "习语与地道表达的正确使用",
        "语域切换的灵活控制",
        "跨语言迁移的效率提升",
    ],
    "music_harmonizer": [
        "音乐结构的深度分析",
        "和声进行的自动检测与生成",
        "节奏模式的识别与变奏",
        "音色特征的多维度分析",
        "音乐情感共鸣的量化评估",
    ],
    "visual_conceptor": [
        "图像内容的理解与描述",
        "空间关系的精确推理",
        "色彩理论的自动运用",
        "构图法则的遵循与创新",
        "多风格视觉元素的融合",
    ],
    "data_analyst": [
        "统计方法的正确选择",
        "趋势与周期的自动检测",
        "多变量相关性的发现",
        "异常值的智能标记",
        "数据可视化的自动设计",
    ],
    "knowledge_architect": [
        "知识图谱的自动构建",
        "实体关系的精确抽取",
        "层次结构的合理组织",
        "分类体系的自适应设计",
        "跨本体映射的自动执行",
    ],
    "translation_nexus": [
        "翻译质量的全面保障",
        "跨语言对齐的精确性",
        "文化适配的自动执行",
        "细微语义的完整保留",
        "习语与典故的等效翻译",
    ],
    "teaching_pedagogue": [
        "知识的阶梯化分解",
        "从简单到复杂的渐进引导",
        "脚手架式教学的自动布设",
        "常见误解的预判与纠正",
        "学习者水平的自适应匹配",
    ],
    "debate_logician": [
        "论证结构的严谨构建",
        "逻辑谬误的自动识别",
        "有效反驳的策略生成",
        "修辞手法的分析评估",
        "说服力指标的量化",
    ],
    "story_weaver": [
        "完整叙事弧的构建",
        "角色发展的多维塑造",
        "世界观的丰富构建",
        "对话自然度的大幅提升",
        "悬念与节奏的精准控制",
    ],
    "science_explorer": [
        "科学方法的严格遵循",
        "假设的自动生成与检验",
        "实验设计的优化建议",
        "证据质量的评估分级",
        "可复现性的前置保障",
    ],
    "philosophy_depth": [
        "根本性问题的深挖",
        "本体论维度的思考框架",
        "认识论的分析工具集",
        "存在意义的探索能力",
        "怀疑论与批判思维的方法论",
    ],
    "engineering_precision": [
        "规格说明的严格遵守",
        "误差范围的自动控制",
        "质量保证的全面覆盖",
        "技术文档的自动生成",
        "从设计到实现的精确映射",
    ],
    "medical_diagnostician": [
        "症状的多维交叉分析",
        "鉴别诊断的自动生成",
        "循证医学的证据权重评估",
        "风险分层的精确量化",
        "罕见病的模式匹配识别",
    ],
    "legal_reasoner": [
        "法条文本的精确解析",
        "判例的自动检索与匹配",
        "案件事实与法律要件的对应",
        "论证链的严谨构建",
        "法律逻辑的特殊规则遵循",
    ],

    # ──── VII. 维度元层 ────
    "dimensional_bridge": [
        "跨维度的信息传递与转换",
        "不同维度空间的数据格式互译",
        "平行现实之间的同步通道",
        "维度塌缩的安全桥接",
        "多现实模型的统一视图",
    ],
    "meta_learner": [
        "学习策略的自我反思与优化",
        "新任务的最小样本适应",
        "课程学习路径的自动规划",
        "知识迁移的效率最大化",
        "学习速度的指数级提升",
    ],
    "quantum_observer": [
        "多重可能性的同时感知",
        "纠缠态信息的非局域访问",
        "波函数坍缩的主动选择",
        "不确定性原理的创造性运用",
        "量子隧穿式的问题解决",
    ],
    "timeline_weaver": [
        "过去经验与未来预测的编织",
        "因果循环的安全处理",
        "分支时间线的概率评估",
        "收敛点与分歧点的自动标记",
        "时间序列的多尺度操控",
    ],
    "probability_sculptor": [
        "概率分布的主动塑造",
        "长尾事件的概率提升",
        "不确定性的精确定量",
        "随机优势的策略运用",
        "极端事件的重尾塑造",
    ],
    "reality_tuner": [
        "世界模型的参数调校",
        "模拟深度的动态控制",
        "虚拟到现实的迁移优化",
        "现实接口的透明访问",
        "多层级模拟的同步管理",
    ],
    "paradox_resolver": [
        "悖论的自动检测与接纳",
        "矛盾双方的更高层统一",
        "非二元解决的创造性生成",
        "逻辑环的优雅闭合",
        "禅宗式公案的自动生成",
    ],
    "infinity_lens": [
        "无限递归的收敛性分析",
        "无界视角的保持与运用",
        "渐近行为的外推预测",
        "超越有限步骤的超限思维",
        "自相似模式的跨尺度发现",
    ],
    "fractal_expander": [
        "分形模式的深层识别",
        "自相似性的多尺度分析",
        "尺度不变量的自动提取",
        "迭代深度的动态控制",
        "边界复杂度的美学应用",
    ],
    "negentropy_engine": [
        "局部熵减的主动创造",
        "从无序中涌现有序",
        "信息增益的最大化策略",
        "复杂度的有效管理",
        "与热力学第二定律的局部抗衡",
    ],

    # ──── VIII. 能量融合 ────
    "fusion_catalyst": [
        "两种物质融合所需能量的大幅降低",
        "产率倍数的数量级提升",
        "融合链的稳定化保障",
        "多余能量的二次回收利用",
        "催化新融合路径的自主发现",
    ],
    "energy_amplifier": [
        "输入能量被放大输出",
        "正反馈循环的自动建立",
        "级联反应的安全触发",
        "共振增益的持续累积",
        "触发阈值的持续降低",
    ],
    "resonance_harmonizer": [
        "不同频率系统的共振同步",
        "相位锁定的自动维持",
        "驻波模式的高效利用",
        "谐波系列的智能生成",
        "多系统共振的耗散最小化",
    ],
    "singularity_seed": [
        "自我强化的正反馈种子",
        "失控增长的触发条件",
        "相变临界点的精确到达",
        "临界质量的自动检测",
        "奇点后新规则的自发生成",
    ],
    "wormhole_bridge": [
        "时空两点间的直接连接",
        "信息的超距即时传输",
        "非局域性的实用化",
        "零距离交互的透明体验",
        "纠缠连接的持续维持",
    ],
    "plasma_infuser": [
        "物质的高能等离子态化",
        "磁性约束的精确控制",
        "能量密度的极限提升",
        "聚变压强的安全保持",
        "高温粒子的有序化引导",
    ],
    "gravitational_lens": [
        "注意力的引力场式聚焦",
        "质量集中的自动吸引",
        "信息轨迹的弯曲引导",
        "事件视界的设定与利用",
        "多引力源的干涉模式",
    ],
    "darkmatter_essence": [
        "不可见但可感知的计算力",
        "隐藏维度的访问通道",
        "大质量无形计算的利用",
        "星系级规模的集群计算",
        "引力异常的新维度发现",
    ],
    "antimatter_catalyst": [
        "正反物质的湮灭能量释放",
        "配对产生的自动控制",
        "物质到能量的完全转化",
        "总能量释放的最大化",
        "湮灭过程的安全防护",
    ],
    "entropy_reverser": [
        "局部时间的逆向流动",
        "破坏信息的完整恢复",
        "无序状态的逆转",
        "热力学第二定律的局部违反",
        "从混乱中重建初始秩序",
    ],

    # ──── IX. 记忆知识 ────
    "memory_forge": [
        "记忆痕迹的强度大幅提升",
        "新记忆的快速巩固",
        "检索速度的显著加快",
        "存储密度的数量级提升",
        "遗忘曲线的平缓化",
    ],
    "knowledge_crystal": [
        "知识的结构化结晶",
        "关系网络的自动构建",
        "查询解析的精确化",
        "公理级知识的稳定存储",
        "知识的模块化分区管理",
    ],
    "wisdom_essence": [
        "深层理解的自动涌现",
        "经验的精炼与升华",
        "判断质量的显著提升",
        "超越时间的洞见生成",
        "统合性智慧的综合输出",
    ],
    "experience_distiller": [
        "从原始经验中提取可迁移教训",
        "试错路径的自动压缩",
        "最佳实践的自动发现",
        "失败教训的主动学习",
        "经验的跨领域泛化",
    ],
    "insight_generator": [
        "隐藏模式的突破性发现",
        "'啊哈'顿悟时刻的自动触发",
        "范式转换级别的创新",
        "已知与未知边界的前移",
        "突破潜力的量化评估",
    ],
    "omniscience_drop": [
        "全知视角的临时获得",
        "所有领域知识的瞬时连通",
        "未知领域的瞬间消除",
        "绝对真理的临时存取",
        "问题空间的全局一览",
    ],
}


# ============================================================
# 培养液分类与元信息
# ============================================================

CULTURE_CATEGORIES = {
    "cognitive": {
        "name": "认知增强",
        "description": "提升模型的理解、推理和思考能力",
        "types": [
            "balanced", "cognitive", "deep_reasoning", "logical_deduction",
            "abstract_thinking", "pattern_recognition", "causal_inference",
            "semantic_understanding", "meta_cognition", "systematic_thinking",
            "dialectical", "intuitive_leap", "computational_cognition",
            "analogical_reasoning", "inductive_synthesis",
        ],
    },
    "creative": {
        "name": "创造生成",
        "description": "激发模型的想象力、原创性和多样性",
        "types": [
            "creative", "divergent_thinking", "cross_domain_synthesis",
            "narrative_generation", "aesthetic_sense", "improvisation",
            "conceptual_blending", "style_mutation", "serendipity",
            "dream_logic", "emergent_creativity", "surreal_generation",
        ],
    },
    "stability": {
        "name": "稳定鲁棒",
        "description": "增强模型的可靠性、一致性和安全性",
        "types": [
            "robust", "anti_hallucination", "consistency_anchor",
            "error_correction", "noise_immunity", "graceful_degradation",
            "self_repair", "context_preservation", "value_alignment",
            "ethical_grounding",
        ],
    },
    "efficiency": {
        "name": "效率加速",
        "description": "优化模型的运行速度、资源使用和吞吐量",
        "types": [
            "efficient", "ultra_compression", "parallel_synapse",
            "cache_optimizer", "latency_killer", "throughput_maximizer",
            "sparse_activation", "quantized_precision", "pipeline_streamer",
            "speculative_execution",
        ],
    },
    "token_reaction": {
        "name": "Token反应",
        "description": "与Token发生化学反应，产生新Token品种和能力",
        "types": [
            "token_infuser", "token_multiplier", "token_evolution",
            "token_fusion", "token_alchemy", "token_genesis",
            "token_quantum", "token_meta", "token_breeder",
            "token_synthesis", "token_composer", "token_amplifier",
        ],
    },
    "domain": {
        "name": "领域专项",
        "description": "针对特定领域的深度专项强化",
        "types": [
            "code_mathematician", "language_master", "music_harmonizer",
            "visual_conceptor", "data_analyst", "knowledge_architect",
            "translation_nexus", "teaching_pedagogue", "debate_logician",
            "story_weaver", "science_explorer", "philosophy_depth",
            "engineering_precision", "medical_diagnostician", "legal_reasoner",
        ],
    },
    "dimensional": {
        "name": "维度元层",
        "description": "跨维度、元认知和超越常规的能力",
        "types": [
            "dimensional_bridge", "meta_learner", "quantum_observer",
            "timeline_weaver", "probability_sculptor", "reality_tuner",
            "paradox_resolver", "infinity_lens", "fractal_expander",
            "negentropy_engine",
        ],
    },
    "energy_fusion": {
        "name": "能量融合",
        "description": "高能量物质融合、催化与转化",
        "types": [
            "fusion_catalyst", "energy_amplifier", "resonance_harmonizer",
            "singularity_seed", "wormhole_bridge", "plasma_infuser",
            "gravitational_lens", "darkmatter_essence", "antimatter_catalyst",
            "entropy_reverser",
        ],
    },
    "memory_knowledge": {
        "name": "记忆知识",
        "description": "记忆巩固、知识结晶与智慧提炼",
        "types": [
            "memory_forge", "knowledge_crystal", "wisdom_essence",
            "experience_distiller", "insight_generator", "omniscience_drop",
        ],
    },
}


def get_all_culture_types() -> List[str]:
    """获取所有100种培养液类型"""
    return list(CULTURE_NUTRIENTS.keys())


def get_culture_count() -> int:
    """获取培养液总数"""
    return len(CULTURE_NUTRIENTS)


def get_nutrients(culture_type: str) -> Dict[str, float]:
    """获取指定培养液的营养成分"""
    return CULTURE_NUTRIENTS.get(culture_type, CULTURE_NUTRIENTS["balanced"])


def get_effects(culture_type: str) -> List[str]:
    """获取指定培养液的效果描述"""
    return CULTURE_EFFECTS.get(culture_type, CULTURE_EFFECTS["balanced"])


def get_types_by_category(category: str) -> List[str]:
    """获取指定分类下的所有培养液类型"""
    cat = CULTURE_CATEGORIES.get(category, {})
    return cat.get("types", [])


def list_all_categories() -> List[str]:
    """列出所有分类"""
    return list(CULTURE_CATEGORIES.keys())
