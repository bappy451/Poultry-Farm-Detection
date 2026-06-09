import torch
import pytest
from poultry_geo.optimizers import build_optimizer, Lion

def test_lion_optimizer_step():
    params = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]
    optimizer = build_optimizer("lion", params, lr=1e-3, weight_decay=0.01)
    
    initial_params = params[0].clone().detach()
    
    # Mock gradient
    params[0].grad = torch.randn(2, 2)
    
    optimizer.step()
    
    assert not torch.equal(params[0], initial_params)
    assert isinstance(optimizer, Lion)

def test_build_optimizer_adan():
    params = [torch.nn.Parameter(torch.randn(2, 2))]
    optimizer = build_optimizer("adan", params, lr=1e-3, weight_decay=0.01)
    from poultry_geo.optimizers import Adan
    assert isinstance(optimizer, Adan)

def test_build_optimizer_invalid():
    params = [torch.nn.Parameter(torch.randn(2, 2))]
    with pytest.raises(ValueError, match="optimizer must be one of"):
        build_optimizer("invalid", params, lr=1e-3, weight_decay=0.01)
