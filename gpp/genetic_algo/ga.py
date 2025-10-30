import random
import numpy as np
from typing import List, Tuple

from .scoring import fitness_function_from_tokens


def genetic_algorithm(
    model,
    device: str,
    x_tokens,
    text_features,
    original_cls,
    num_patches: int,
    keep: int,
    population_size: int = 20,
    generations: int = 30,
    mutation_rate: float = 0.1,
) -> List[int]:
    """Simple GA for selecting a fixed-size subset of patch indices to keep.

    Uses cached pre-LN tokens (`x_tokens`) for fast fitness and treats
    individuals as kept-index lists; masking uses the complement as
    indices-to-remove for fitness evaluation.
    """
    population = [random.sample(range(num_patches), keep) for _ in range(population_size)]

    def score_kept(kept_idx: List[int]) -> float:
        # convert kept indices to indices-to-remove
        kept_set = set(kept_idx)
        indices_to_remove = [i for i in range(num_patches) if i not in kept_set]
        return fitness_function_from_tokens(
            model, device, x_tokens, indices_to_remove, text_features, original_cls
        )

    for _ in range(generations):
        scores = [score_kept(ind) for ind in population]
        total = sum(scores) or 1.0
        probs = [s / total for s in scores]

        selected = np.random.choice(population_size, size=population_size, p=probs)
        parents = [population[i] for i in selected]

        next_gen = []
        for i in range(0, population_size, 2):
            p1, p2 = parents[i], parents[(i + 1) % population_size]
            cp = random.randint(1, keep - 1)
            children = [p1[:cp] + p2[cp:], p2[:cp] + p1[cp:]]

            for child in children:
                if random.random() < mutation_rate:
                    child[random.randrange(keep)] = random.randrange(num_patches)
                # repair: unique, trim, and pad to keep length
                child = sorted(set(child))[:keep]
                while len(child) < keep:
                    new_patch = random.randrange(num_patches)
                    if new_patch not in child:
                        child.append(new_patch)
                next_gen.append(child)

        population = next_gen

    best = max(population, key=score_kept)
    return best


def _repair_bits(bits, min_keep: int, max_keep: int):
    """Ensure number of ones lies within [min_keep, max_keep] by flipping bits."""
    import random

    ones_idx = [i for i, b in enumerate(bits) if b == 1]
    zeros_idx = [i for i, b in enumerate(bits) if b == 0]
    keep = len(ones_idx)

    if keep < min_keep:
        need = min_keep - keep
        flip = random.sample(zeros_idx, k=min(need, len(zeros_idx))) if zeros_idx else []
        for i in flip:
            bits[i] = 1
    elif keep > max_keep:
        need = keep - max_keep
        flip = random.sample(ones_idx, k=min(need, len(ones_idx))) if ones_idx else []
        for i in flip:
            bits[i] = 0
    return bits


def _bits_to_indices(bits):
    return [i for i, b in enumerate(bits) if b == 1]


def genetic_algorithm_variable_keep(
    model,
    device: str,
    x_tokens,
    text_features,
    original_cls,
    num_patches: int,
    min_keep: int,
    max_keep: int,
    population_size: int = 30,
    generations: int = 40,
    mutation_rate: float = 0.1,
    bit_flip_rate: float = 0.05,
    keep_penalty: float = 0.1,
) -> Tuple[List[int], float]:
    """GA over a binary chromosome (length=num_patches) to choose which patches to keep.

    - 1 bit means keep, 0 means remove.
    - A small penalty encourages fewer kept patches (trade-off with accuracy).
    - Repair ensures kept count stays within [min_keep, max_keep].
    Returns list of kept indices.
    """
    import random

    def score_bits(bits):
        kept = sum(bits)
        # indices to remove are zeros
        indices_to_remove = [i for i, b in enumerate(bits) if b == 0]
        base = fitness_function_from_tokens(
            model, device, x_tokens, indices_to_remove, text_features, original_cls
        )
        penalty = keep_penalty * (kept / num_patches)
        return max(0,base - penalty)

    # Initialize population with random feasible bitstrings within [min_keep, max_keep]
    population = []
    for _ in range(population_size):
        # random ratio within bounds
        r = random.uniform(min_keep / num_patches, max_keep / num_patches, )
        r = round(r, 2)
        bits = [1 if random.random() < r else 0 for _ in range(num_patches)]
        # print(xf'{bits = }')
        bits = _repair_bits(bits, min_keep, max_keep)
        population.append(bits)

    for _ in range(generations):
        scores = [score_bits(bits) for bits in population]
        total = sum(scores) or 1.0
        probs = [s / total for s in scores]

        selected = np.random.choice(population_size, size=population_size, p=probs)
        parents = [population[i] for i in selected]

        next_gen = []
        for i in range(0, population_size, 2):
            p1, p2 = parents[i], parents[(i + 1) % population_size]
            # single-point crossover over bits
            cp = random.randint(1, num_patches - 1)
            c1 = p1[:cp] + p2[cp:]
            c2 = p2[:cp] + p1[cp:]

            for child in (c1, c2):
                # mutate: flip random bits with bit_flip_rate; also rare reassign via mutation_rate
                for j in range(num_patches):
                    if random.random() < bit_flip_rate:
                        child[j] = 1 - child[j]
                if random.random() < mutation_rate:
                    j = random.randrange(num_patches)
                    child[j] = 1 - child[j]
                child = _repair_bits(child, min_keep, max_keep)
                next_gen.append(child)

        population = next_gen

    best = max(population, key=lambda bits: score_bits(bits))
    indices = _bits_to_indices(best)
    keep_pct = (len(indices) / num_patches) if num_patches > 0 else 0.0
    keep_pct = round(keep_pct, 2)
    return indices, keep_pct
