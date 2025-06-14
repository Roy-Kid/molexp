import pytest
from molexp.param import ParamSpace, CartesianSampler, RandomSampler, CombinationSampler

def test_cartesian_sampler():
    space = ParamSpace({"x": [1, 2], "y": ["a", "b"]})
    sampler = CartesianSampler()
    samples = list(sampler.sample(space))
    assert len(samples) == 4
    assert {tuple(sample.items()) for sample in samples} == {
        (("x", 1), ("y", "a")),
        (("x", 1), ("y", "b")),
        (("x", 2), ("y", "a")),
        (("x", 2), ("y", "b")),
    }

def test_random_sampler():
    space = ParamSpace({"x": [1, 2], "y": ["a", "b"]})
    sampler = RandomSampler(num_samples=5)
    samples = list(sampler.sample(space))
    assert len(samples) == 5
    for sample in samples:
        assert sample["x"] in [1, 2]
        assert sample["y"] in ["a", "b"]

def test_combination_sampler():
    space = ParamSpace({"x": [1, 2], "y": ["a", "b"]})
    sampler = CombinationSampler(r=2)
    samples = list(sampler.sample(space))
    assert len(samples) == 6  # Combinations of 2 from Cartesian product
    for sample in samples:
        assert len(sample) == 2
