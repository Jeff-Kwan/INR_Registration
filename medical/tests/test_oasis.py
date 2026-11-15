"""Run shared synthetic metric-parity and mocked adapter checks on OASIS/35 labels."""
from medical.tests import test_abdomen as template

class TestOASIS(template.TestAbdomen):
    dataset="oasis"
