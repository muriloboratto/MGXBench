"""Unit tests for weak-scaling size alignment and campaign configuration."""
import importlib.util
from pathlib import Path
import pytest

SPEC = importlib.util.spec_from_file_location(
    'run_campaign', Path(__file__).resolve().parents[1] / 'configuration' / 'run_campaign.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)

@pytest.mark.parametrize('ranks,expected', [(1,4096),(4,6500),(8,8192)])
def test_weak_scaling_1d(ranks, expected):
    theoretical, actual = module.weak_scaling_size(4096, ranks, '1d')
    assert actual == expected
    assert actual % 4 == 0
    assert actual % ranks == 0
    assert abs((actual / 4096) ** 3 / ranks - 1) < .025

@pytest.mark.parametrize('ranks', [1,4,9,16])
def test_weak_scaling_2d_square_grid(ranks):
    _, n = module.weak_scaling_size(4096, ranks, '2d')
    assert n % 4 == 0
    assert n % int(ranks ** .5) == 0


def test_weak_scaling_2d_rejects_non_square():
    with pytest.raises(ValueError, match='square process grid'):
        module.weak_scaling_size(4096, 8, '2d')


def test_campaign_skips_2d_eight_ranks():
    configs = module.configurations(['1d','2d'], [4096], [1,4,8], 'weak', 4096)
    assert ('1d',6500,4) in configs
    assert all(not (d == '2d' and p == 8) for d,n,p in configs)


def test_strong_mode_keeps_fixed_size():
    configs = module.configurations(['1d'], [4096], [1,4,8], 'strong', 4096)
    assert [n for _,n,_ in configs] == [4096,4096,4096]


def test_custom_alignment():
    _, n = module.weak_scaling_size(4096, 4, '1d', 16)
    assert n % 16 == 0


@pytest.mark.parametrize('base,ranks,alignment', [(0,4,4),(4096,0,4),(4096,4,0)])
def test_invalid_values(base,ranks,alignment):
    with pytest.raises(ValueError):
        module.weak_scaling_size(base,ranks,'1d',alignment)
