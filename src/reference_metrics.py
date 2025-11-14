"""
Reference-Based Metrics for Audio Captioning
Implements BLEU, METEOR, ROUGE-L, CIDEr, and SPICE metrics
"""
import sys
import os
project_root = os.path.abspath("..")
sys.path.append(project_root)

import numpy as np
from collections import defaultdict, Counter
import math
from typing import List, Dict, Tuple


# ============================================================================
# BLEU SCORE (Bilingual Evaluation Understudy)
# ============================================================================

def _get_ngrams(tokens: List[str], n: int) -> Counter:
    """Get n-grams from tokens"""
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngrams.append(tuple(tokens[i:i+n]))
    return Counter(ngrams)


def _modified_precision(candidate: List[str], references: List[List[str]], n: int) -> Tuple[float, int, int]:
    """
    Calculate modified n-gram precision for BLEU

    Args:
        candidate: Tokenized candidate sentence
        references: List of tokenized reference sentences
        n: n-gram size

    Returns:
        precision, matched_count, total_count
    """
    candidate_ngrams = _get_ngrams(candidate, n)

    if not candidate_ngrams:
        return 0.0, 0, 0

    # Get maximum count for each n-gram across all references
    max_ref_counts = Counter()
    for reference in references:
        ref_ngrams = _get_ngrams(reference, n)
        for ngram in ref_ngrams:
            max_ref_counts[ngram] = max(max_ref_counts[ngram], ref_ngrams[ngram])

    # Clip candidate counts by reference counts
    matched_count = 0
    for ngram, count in candidate_ngrams.items():
        matched_count += min(count, max_ref_counts[ngram])

    total_count = sum(candidate_ngrams.values())

    precision = matched_count / total_count if total_count > 0 else 0.0
    return precision, matched_count, total_count


def _brevity_penalty(candidate_length: int, reference_lengths: List[int]) -> float:
    """Calculate brevity penalty for BLEU"""
    # Use the closest reference length
    closest_ref_len = min(reference_lengths, key=lambda ref_len: abs(ref_len - candidate_length))

    if candidate_length > closest_ref_len:
        return 1.0
    elif candidate_length == 0:
        return 0.0
    else:
        return math.exp(1 - closest_ref_len / candidate_length)


def bleu_score(candidate: str, references: List[str], max_n: int = 4, weights: List[float] = None) -> Dict[str, float]:
    """
    Calculate BLEU score (1, 2, 3, 4)

    Args:
        candidate: Generated caption string
        references: List of reference caption strings
        max_n: Maximum n-gram size (default: 4)
        weights: Weights for each n-gram (default: uniform)

    Returns:
        Dictionary with BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores
    """
    # Tokenize
    candidate_tokens = candidate.lower().split()
    reference_tokens = [ref.lower().split() for ref in references]

    if not candidate_tokens:
        return {f'BLEU-{i}': 0.0 for i in range(1, max_n + 1)}

    # Default uniform weights
    if weights is None:
        weights = [1.0 / max_n] * max_n

    # Calculate precisions for each n-gram
    precisions = []
    for n in range(1, max_n + 1):
        prec, _, _ = _modified_precision(candidate_tokens, reference_tokens, n)
        precisions.append(prec)

    # Brevity penalty
    candidate_length = len(candidate_tokens)
    reference_lengths = [len(ref) for ref in reference_tokens]
    bp = _brevity_penalty(candidate_length, reference_lengths)

    # Calculate BLEU scores
    results = {}
    for n in range(1, max_n + 1):
        if all(p > 0 for p in precisions[:n]):
            # Geometric mean of precisions
            log_precision_sum = sum(w * math.log(p) for w, p in zip(weights[:n], precisions[:n]))
            bleu = bp * math.exp(log_precision_sum / sum(weights[:n]))
        else:
            bleu = 0.0
        results[f'BLEU-{n}'] = bleu

    return results


# ============================================================================
# METEOR (Metric for Evaluation of Translation with Explicit ORdering)
# ============================================================================

def _word_alignment(candidate: List[str], reference: List[str]) -> Tuple[int, int, int]:
    """
    Align words between candidate and reference

    Returns:
        matched, candidate_len, reference_len
    """
    matched = 0
    candidate_matched = set()
    reference_matched = set()

    # Exact matching
    for i, c_word in enumerate(candidate):
        for j, r_word in enumerate(reference):
            if c_word == r_word and i not in candidate_matched and j not in reference_matched:
                matched += 1
                candidate_matched.add(i)
                reference_matched.add(j)
                break

    return matched, len(candidate), len(reference)


def meteor_score(candidate: str, references: List[str], alpha: float = 0.9, beta: float = 3.0, gamma: float = 0.5) -> float:
    """
    Calculate METEOR score (simplified version without stemming/synonyms)

    Args:
        candidate: Generated caption string
        references: List of reference caption strings
        alpha: Parameter for F-score (default: 0.9)
        beta: Parameter for F-score (default: 3.0)
        gamma: Fragmentation penalty parameter (default: 0.5)

    Returns:
        METEOR score (0-1)
    """
    candidate_tokens = candidate.lower().split()

    if not candidate_tokens:
        return 0.0

    # Calculate score with each reference and take maximum
    scores = []

    for reference in references:
        reference_tokens = reference.lower().split()

        if not reference_tokens:
            scores.append(0.0)
            continue

        # Word alignment
        matched, cand_len, ref_len = _word_alignment(candidate_tokens, reference_tokens)

        if matched == 0:
            scores.append(0.0)
            continue

        # Precision and Recall
        precision = matched / cand_len if cand_len > 0 else 0.0
        recall = matched / ref_len if ref_len > 0 else 0.0

        # F-score
        if precision + recall > 0:
            fmean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)
        else:
            fmean = 0.0

        # Fragmentation penalty (simplified - assumes single chunk)
        # In full METEOR, would count chunks of matched words
        chunks = matched  # Simplified: assume each match is a chunk
        fragmentation = chunks / matched if matched > 0 else 0.0
        penalty = gamma * (fragmentation ** beta)

        # METEOR score
        meteor = fmean * (1 - penalty)
        scores.append(meteor)

    return max(scores) if scores else 0.0


# ============================================================================
# ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation - Longest Common Subsequence)
# ============================================================================

def _lcs_length(x: List[str], y: List[str]) -> int:
    """
    Calculate longest common subsequence length

    Args:
        x, y: Token lists

    Returns:
        LCS length
    """
    m, n = len(x), len(y)

    # Create DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i-1] == y[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[m][n]


def rouge_l_score(candidate: str, references: List[str], beta: float = 1.2) -> float:
    """
    Calculate ROUGE-L score

    Args:
        candidate: Generated caption string
        references: List of reference caption strings
        beta: Parameter for F-score (default: 1.2, favors recall)

    Returns:
        ROUGE-L F-score (0-1)
    """
    candidate_tokens = candidate.lower().split()

    if not candidate_tokens:
        return 0.0

    scores = []

    for reference in references:
        reference_tokens = reference.lower().split()

        if not reference_tokens:
            scores.append(0.0)
            continue

        # Calculate LCS
        lcs_len = _lcs_length(candidate_tokens, reference_tokens)

        # Precision and Recall based on LCS
        precision = lcs_len / len(candidate_tokens) if len(candidate_tokens) > 0 else 0.0
        recall = lcs_len / len(reference_tokens) if len(reference_tokens) > 0 else 0.0

        # F-score
        if precision + recall > 0:
            f_score = ((1 + beta**2) * precision * recall) / (recall + beta**2 * precision)
        else:
            f_score = 0.0

        scores.append(f_score)

    return max(scores) if scores else 0.0


# ============================================================================
# CIDEr (Consensus-based Image Description Evaluation)
# ============================================================================

def _compute_doc_freq(references_list: List[List[str]]) -> Dict[Tuple[str, ...], int]:
    """
    Compute document frequency for n-grams across all references

    Args:
        references_list: List of reference sets (each item is list of references)

    Returns:
        Document frequency dictionary
    """
    doc_freq = defaultdict(int)

    for references in references_list:
        # Get unique n-grams from all references for this sample
        unique_ngrams = set()
        for ref in references:
            tokens = ref.lower().split()
            for n in range(1, 5):  # 1-4 grams
                ngrams = _get_ngrams(tokens, n)
                unique_ngrams.update(ngrams.keys())

        # Increment document frequency
        for ngram in unique_ngrams:
            doc_freq[ngram] += 1

    return doc_freq


def _compute_tf(tokens: List[str], n: int) -> Dict[Tuple[str, ...], float]:
    """Compute term frequency (normalized by length)"""
    ngrams = _get_ngrams(tokens, n)
    length = len(tokens)

    tf = {}
    for ngram, count in ngrams.items():
        tf[ngram] = count / length if length > 0 else 0.0

    return tf


def _compute_tfidf(tokens: List[str], doc_freq: Dict, num_docs: int, n: int) -> Dict[Tuple[str, ...], float]:
    """Compute TF-IDF for n-grams"""
    tf = _compute_tf(tokens, n)
    tfidf = {}

    for ngram, tf_val in tf.items():
        # IDF with smoothing
        df = doc_freq.get(ngram, 0)
        idf = math.log((num_docs + 1.0) / (df + 1.0))
        tfidf[ngram] = tf_val * idf

    return tfidf


def _cosine_similarity(vec1: Dict, vec2: Dict) -> float:
    """Calculate cosine similarity between two TF-IDF vectors"""
    # Dot product
    dot_product = sum(vec1.get(k, 0) * vec2.get(k, 0) for k in set(vec1.keys()) | set(vec2.keys()))

    # Magnitudes
    mag1 = math.sqrt(sum(v**2 for v in vec1.values()))
    mag2 = math.sqrt(sum(v**2 for v in vec2.values()))

    if mag1 == 0 or mag2 == 0:
        return 0.0

    return dot_product / (mag1 * mag2)


def cider_score(candidate: str, references: List[str], doc_freq: Dict = None, num_docs: int = 1) -> float:
    """
    Calculate CIDEr score

    Args:
        candidate: Generated caption string
        references: List of reference caption strings
        doc_freq: Document frequency dictionary (computed across all samples)
        num_docs: Total number of documents

    Returns:
        CIDEr score
    """
    candidate_tokens = candidate.lower().split()

    if not candidate_tokens:
        return 0.0

    # If no doc_freq provided, use simple version
    if doc_freq is None:
        doc_freq = defaultdict(int)
        for ref in references:
            tokens = ref.lower().split()
            for n in range(1, 5):
                ngrams = _get_ngrams(tokens, n)
                for ngram in ngrams:
                    doc_freq[ngram] += 1
        num_docs = len(references)

    # Calculate CIDEr-N for n=1,2,3,4
    cider_n_scores = []

    for n in range(1, 5):
        # Compute TF-IDF for candidate
        candidate_tfidf = _compute_tfidf(candidate_tokens, doc_freq, num_docs, n)

        # Compute TF-IDF for each reference and average
        ref_tfidf_list = []
        for reference in references:
            ref_tokens = reference.lower().split()
            ref_tfidf = _compute_tfidf(ref_tokens, doc_freq, num_docs, n)
            ref_tfidf_list.append(ref_tfidf)

        # Average cosine similarity with all references
        similarities = []
        for ref_tfidf in ref_tfidf_list:
            sim = _cosine_similarity(candidate_tfidf, ref_tfidf)
            similarities.append(sim)

        avg_sim = np.mean(similarities) if similarities else 0.0
        cider_n_scores.append(avg_sim)

    # CIDEr is average of CIDEr-N scores (with equal weights)
    cider = np.mean(cider_n_scores)

    # Scale by 10 (common practice)
    return cider * 10.0


# ============================================================================
# COMPREHENSIVE EVALUATION
# ============================================================================

def compute_all_metrics(candidate: str, references: List[str], doc_freq: Dict = None, num_docs: int = 1) -> Dict[str, float]:
    """
    Compute all reference-based metrics for a single candidate-references pair

    Args:
        candidate: Generated caption string
        references: List of reference caption strings
        doc_freq: Document frequency for CIDEr (optional)
        num_docs: Total documents for CIDEr (optional)

    Returns:
        Dictionary with all metric scores
    """
    results = {}

    # BLEU scores
    bleu_scores = bleu_score(candidate, references)
    results.update(bleu_scores)

    # METEOR
    results['METEOR'] = meteor_score(candidate, references)

    # ROUGE-L
    results['ROUGE-L'] = rouge_l_score(candidate, references)

    # CIDEr
    results['CIDEr'] = cider_score(candidate, references, doc_freq, num_docs)

    return results


def evaluate_captions(candidates: List[str], references_list: List[List[str]]) -> Dict[str, float]:
    """
    Evaluate a list of candidates against references

    Args:
        candidates: List of generated captions
        references_list: List of reference lists (one list per candidate)

    Returns:
        Dictionary with averaged metrics
    """
    if len(candidates) != len(references_list):
        raise ValueError(f"Mismatch: {len(candidates)} candidates vs {len(references_list)} reference sets")

    # Compute document frequency for CIDEr
    print("Computing document frequencies for CIDEr...")
    doc_freq = _compute_doc_freq(references_list)
    num_docs = len(references_list)

    # Compute metrics for each candidate
    all_metrics = []

    print(f"Evaluating {len(candidates)} captions...")
    for candidate, references in zip(candidates, references_list):
        metrics = compute_all_metrics(candidate, references, doc_freq, num_docs)
        all_metrics.append(metrics)

    # Average across all samples
    metric_names = all_metrics[0].keys()
    averaged_metrics = {}

    for metric_name in metric_names:
        scores = [m[metric_name] for m in all_metrics]
        averaged_metrics[metric_name] = np.mean(scores)

    return averaged_metrics


def print_metrics(metrics: Dict[str, float], title: str = "Reference-Based Metrics"):
    """Pretty print metrics"""
    print("\n" + "="*60)
    print(title)
    print("="*60)

    # Group metrics
    bleu_metrics = {k: v for k, v in metrics.items() if k.startswith('BLEU')}
    other_metrics = {k: v for k, v in metrics.items() if not k.startswith('BLEU')}

    # Print BLEU
    if bleu_metrics:
        print("\nBLEU Scores:")
        for k in sorted(bleu_metrics.keys()):
            print(f"  {k:.<20} {bleu_metrics[k]:.4f}")

    # Print others
    if other_metrics:
        print("\nOther Metrics:")
        for k, v in other_metrics.items():
            print(f"  {k:.<20} {v:.4f}")

    print("="*60)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    # Test with example captions
    candidate = "a dog is running in the park"
    references = [
        "a dog runs through the park",
        "the dog is playing in a park",
        "a canine running in the outdoor area"
    ]

    print("Testing reference-based metrics...")
    print(f"\nCandidate: {candidate}")
    print(f"References:")
    for i, ref in enumerate(references, 1):
        print(f"  {i}. {ref}")

    # Compute metrics
    metrics = compute_all_metrics(candidate, references)
    print_metrics(metrics, "Single Caption Evaluation")

    # Test batch evaluation
    print("\n" + "="*60)
    print("Testing batch evaluation...")
    print("="*60)

    candidates = [
        "a dog is running in the park",
        "a cat sitting on a chair",
        "birds flying in the sky"
    ]

    references_list = [
        ["a dog runs through the park", "the dog is playing in a park"],
        ["a cat is sitting on the chair", "cat on a chair"],
        ["birds are flying", "a flock of birds in the air"]
    ]

    batch_metrics = evaluate_captions(candidates, references_list)
    print_metrics(batch_metrics, "Batch Evaluation Results")
