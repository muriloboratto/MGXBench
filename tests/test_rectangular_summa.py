"""Rectangular SUMMA grid and campaign integration (no GPU required)."""
import importlib.util
from pathlib import Path
import pytest
from configuration.grid import resolve_grid

SPEC = importlib.util.spec_from_file_location('campaign_rect', Path(__file__).resolve().parents[1] / 'configuration' / 'run_campaign.py')
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)

@pytest.mark.parametrize('ranks,grid', [(1,(1,1)),(4,(2,2)),(8,(2,4)),(12,(3,4)),(16,(4,4))])
def test_auto_grid(ranks,grid):
    assert resolve_grid(ranks)==grid

@pytest.mark.parametrize('ranks,rows,cols', [(8,2,4),(8,4,2),(16,4,4),(16,2,8)])
def test_explicit_grid(ranks,rows,cols):
    assert resolve_grid(ranks,rows,cols)==(rows,cols)

@pytest.mark.parametrize('ranks,rows,cols', [(8,3,4),(16,2,4),(8,0,8)])
def test_invalid_grid(ranks,rows,cols):
    with pytest.raises(ValueError):resolve_grid(ranks,rows,cols)

@pytest.mark.parametrize('ranks', [4,8,12,16])
def test_weak_scaling_rectangular(ranks):
    _,n=campaign.weak_scaling_size(4096,ranks,'2d')
    r,c=resolve_grid(ranks)
    assert n%r==0 and n%c==0

@pytest.mark.parametrize('ranks', [4,8,12,16])
def test_strong_campaign_rectangular(ranks):
    assert ('2d',12288,ranks) in campaign.configurations(['2d'],[12288],[ranks],'strong',4096)

def test_2x4_manual_campaign():
    assert campaign.configurations(['2d'],[4096],[8],'sweep',4096,grid_rows=2,grid_cols=4)==[('2d',4096,8)]
