import pytest
import numpy as np
from causalml.dataset import make_mab_data
from causalml.optimize import EpsilonGreedy, LinUCB


def test_make_mab_data():
    df = make_mab_data(n_samples=10, n_arms=2, n_features=2, random_seed=123)
    assert not df.empty
    assert 'reward' in df.columns
    assert 'arm' in df.columns
    assert any(col.startswith('feature_') for col in df.columns)


def test_epsilon_greedy_basic():
    df = make_mab_data(n_samples=10, n_arms=2, n_features=2, random_seed=123)
    algo = EpsilonGreedy(df, reward='reward', arm='arm', epsilon=0.1)
    arm = algo.select_arm()
    assert arm in ['arm_0', 'arm_1']
    algo.update(arm, 1)


def test_linucb_basic():
    df = make_mab_data(n_samples=10, n_arms=2, n_features=2, random_seed=123)
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    algo = LinUCB(df, features=feature_cols, reward='reward', arm='arm', alpha=1.0)
    context = df.iloc[0][feature_cols].values
    arm = algo.select_arm(context)
    assert arm in ['arm_0', 'arm_1']
    algo.update(arm, context, 1) 