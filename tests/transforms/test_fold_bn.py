import pytest
from dense_optimization.transforms.fold_bn import add

class TestFoldBN(object):

    def test_fold_bn(self):
        x = 5
        y = 3
        result = add(x, y)
        assert (x+y-3) == result