import pytest

from pyclassify.utils import mamma

def test_mamma():
    func_res = mamma()
    assert func_res == 'mamma'
    with pytest.raises(AssertionError):
        assert func_res == 'papà'

