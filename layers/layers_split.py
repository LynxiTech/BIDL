# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.


import torch as pt

from .lif import ConvLif2d, Lif2d
from .temporal_aggregation import Aggregation
from .thresh_firing import ThreshFiring
from .time_distributed import TimeDistributed


class ThreshFiringSplit(ThreshFiring):

    def forward(self, xis: list) -> list:
        assert isinstance(xis, (list, tuple))
        xos = [super(ThreshFiringSplit, self).forward(_) for _ in xis]
        return xos


class ConvLif2dSplit(ConvLif2d):
    """Time-split version of ``ConvLif2d``."""

    def forward(self, xis: list):
        assert isinstance(xis, (list, tuple))
        t = len(xis)
        hx = None
        xo_list = []
        for i in range(t):
            xi_t = xis[i]
            xo_t, hx = self.cell(xi_t, hx)
            xo_list.append(xo_t)
        if self.ret_hx:
            return xo_list, hx
        return xo_list


class Lif2dSplit(Lif2d):
    """Time-split version pf ``Lif2d``."""

    def forward(self, xis: list):
        assert isinstance(xis, (list, tuple))
        t = len(xis)
        hx = None
        xo_list = []
        for i in range(t):
            xi_t = xis[i]
            xo_t, hx = self.cell(xi_t, hx)
            xo_list.append(xo_t)
        if self.ret_hx:
            return xo_list, hx
        return xo_list


class TimeDistributedSplit(TimeDistributed):
    """Time-split version of ``TimeDistributed``."""

    def forward(self, xis: list) -> list:
        """
        :param xis: of shape ``[(b, c, h, w), ...]``
        :return: ``xos``, of shape ``[(b, c, h, w), ...]``
        """
        assert isinstance(xis, (list, tuple))
        xos = [self.module(_) for _ in xis]
        return xos


class FlattenSplit(pt.nn.Flatten):
    """Time-split version of ``Flatten``."""

    def __init__(self, start_dim=1, end_dim=-1):
        assert start_dim in [1, 2] and end_dim == -1  # XXX
        super(FlattenSplit, self).__init__(start_dim, end_dim)

    def forward(self, xis: list) -> pt.Tensor:
        """
        :param xis: of shape ``[(b, c, ...), ...]``
        :return: ``xos``, of shape ``(b, ...)``
        """
        if isinstance(xis, pt.Tensor):  # This means there's an ``Aggregation`` before this layer.
            return super(FlattenSplit, self).forward(xis)
        if xis[0].ndim > 3:
            xis = [pt.flatten(_, 2, -1) for _ in xis]  # [(b, c, ?)]
        xis2 = pt.stack(xis, dim=1)  # (b, t, c, ?)
        xos = super(FlattenSplit, self).forward(xis2)
        return xos


class AggregationSplit(Aggregation):
    """Time-split version of ``Aggregation``."""

    def __init__(self, mode='mean', dim=1, idx=-1):
        assert dim == 1  # XXX
        super(AggregationSplit, self).__init__(mode, dim, idx)

    def forward(self, xis: list) -> list:
        if isinstance(xis, pt.Tensor):
            return super(AggregationSplit, self).forward(xis)
        if self.mode == 'sum':
            return sum(xis)
        elif self.mode == 'mean':
            return sum(xis) / len(xis)
        elif self.mode == 'pick':
            return xis[self.idx]
        else:
            raise NotImplementedError
