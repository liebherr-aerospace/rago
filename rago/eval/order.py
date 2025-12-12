"""Define the Oder class containing the order with which to present the candidates in pairwise evaluation."""

from enum import StrEnum


class Order(StrEnum):
    """The order in which the judge will see the answers when evaluating pairwise.

    SAME: The order is the same.
    REVERSED: The order is reversed.
    """

    SAME = "same"
    """
    The order is the same.
    """
    REVERSED = "reversed"
    """
    The order is reversed
    """
