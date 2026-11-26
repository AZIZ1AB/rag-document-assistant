def precision_at_k(retrieved_ids, relevant_ids, k):
    """Fraction of the first k retrieved IDs that are relevant."""
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    return sum(item in relevant_ids for item in top_k) / len(top_k)


def recall_at_k(retrieved_ids, relevant_ids, k):
    """Fraction of all known relevant IDs found in the first k results."""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    return sum(item in relevant_ids for item in top_k) / len(relevant_ids)


def reciprocal_rank(retrieved_ids, relevant_ids):
    """Reciprocal rank of the first relevant result (zero if none is found)."""
    for rank, item in enumerate(retrieved_ids, start=1):
        if item in relevant_ids:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(all_retrieved_ids, all_relevant_ids):
    """Average reciprocal rank over a collection of evaluation questions."""
    if not all_retrieved_ids:
        return 0.0
    scores = [
        reciprocal_rank(retrieved, relevant)
        for retrieved, relevant in zip(all_retrieved_ids, all_relevant_ids)
    ]
    return sum(scores) / len(scores)
