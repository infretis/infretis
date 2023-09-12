"""Some external order parameters for infretis."""
from infretis.classes.orderparameter import OrderParameter


class ConstantOrder(OrderParameter):
    """A constant order parameter for testing."""

    def __init__(self, constant):
        super().__init__(description="A constant order parameter.")
        self.constant = constant

    def calculate(self, system):
        return self.constant


class OrderMissingCalculate(OrderParameter):
    """An order parameter without the calculate method."""

    calculate = 123
