import pytest
@pytest.mark.parametrize("qty,price,valid", [(10,280,True),(1000,280,False)])
def test_position_limit(qty,price,valid):
    assert (qty*price <= 100000) == valid

@pytest.mark.parametrize("conf,valid", [(0.8,True),(0.3,False)])  
def test_confidence(conf,valid):
    assert (conf >= 0.6) == valid
