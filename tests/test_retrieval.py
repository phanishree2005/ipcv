"""
Unit tests for FAISS retrieval index, metadata search, and multi-context re-ranking.
"""

import numpy as np
import pytest
from src.retrieval.faiss_index import ReIDFAISSIndex
from src.retrieval.reranking import MultiContextReRanker, k_reciprocal_reranking

def test_faiss_index_and_search():
    index = ReIDFAISSIndex(embedding_dim=512)
    assert index.total_entries == 0

    # Add 10 gallery samples
    np.random.seed(42)
    for i in range(10):
        emb = np.random.randn(512).astype(np.float32)
        index.add_gallery_entry(
            embedding=emb,
            global_id=f"P{i%3:03d}",
            camera_id=i % 4 + 1,
            timestamp=100.0 + i * 15,
            track_id=i,
            quality_score=0.85
        )

    assert index.total_entries == 10

    # Search
    q_emb = np.random.randn(512).astype(np.float32)
    results = index.search(q_emb, top_k=5)
    assert len(results) == 5
    assert results[0]["rank"] == 1
    assert "similarity" in results[0]
    assert "global_id" in results[0]

def test_reranking():
    reranker = MultiContextReRanker()
    query_info = {"camera_id": 1, "timestamp": 100.0, "pose_vector": np.zeros(34)}
    candidates = [
        {"similarity": 0.85, "camera_id": 2, "timestamp": 135.0, "quality_score": 0.9, "pose_vector": np.zeros(34)},
        {"similarity": 0.88, "camera_id": 6, "timestamp": 101.0, "quality_score": 0.6, "pose_vector": np.zeros(34)},
    ]

    reranked = reranker.rerank_candidates(query_info, candidates)
    assert len(reranked) == 2
    assert "final_score" in reranked[0]
    assert reranked[0]["reranked_rank"] == 1
