import torch

from model_core.formula_simplifier import formula_to_canonical_key, simplify_formula
from model_core.vm import StackVM


def test_simplify_formula_collapses_redundant_real_world_case():
    formula = [
        "REMAIN_SIZE",
        "PURE_VALUE",
        "PREM",
        "PURE_VALUE",
        "PURE_VALUE",
        "PURE_VALUE",
        "MUL",
        "ABS",
        "TURNOVER",
        "SQRT",
        "ADD",
        "MIN",
        "MIN",
        "MAX",
        "MIN",
    ]

    simplified = simplify_formula(formula)
    assert simplified == ["PURE_VALUE", "REMAIN_SIZE", "MIN"]


def test_formula_to_canonical_key_normalizes_commutative_equivalence():
    left = ["PURE_VALUE", "REMAIN_SIZE", "MIN"]
    right = ["REMAIN_SIZE", "PURE_VALUE", "MIN"]

    assert formula_to_canonical_key(left) == formula_to_canonical_key(right)


def test_simplify_formula_applies_absorption_rule():
    formula = ["PURE_VALUE", "PREM", "PURE_VALUE", "MIN", "MAX"]

    simplified = simplify_formula(formula)
    assert simplified == ["PURE_VALUE"]


def test_simplified_formula_matches_original_vm_output(monkeypatch):
    monkeypatch.setattr(
        "model_core.config.ModelConfig.INPUT_FEATURES",
        ["REMAIN_SIZE", "PURE_VALUE", "PREM", "TURNOVER"],
        raising=False,
    )

    formula = [
        "REMAIN_SIZE",
        "PURE_VALUE",
        "PREM",
        "PURE_VALUE",
        "PURE_VALUE",
        "PURE_VALUE",
        "MUL",
        "ABS",
        "TURNOVER",
        "SQRT",
        "ADD",
        "MIN",
        "MIN",
        "MAX",
        "MIN",
    ]
    simplified = simplify_formula(formula)

    feat_tensor = torch.tensor(
        [
            [[10.0, 8.0, 12.0, 4.0], [15.0, 20.0, 5.0, 9.0]],
            [[11.0, 9.0, 3.0, 16.0], [14.0, 18.0, 7.0, 25.0]],
        ],
        dtype=torch.float32,
    )

    vm = StackVM()
    original_out = vm.execute(formula, feat_tensor)
    simplified_out = vm.execute(simplified, feat_tensor)

    assert original_out is not None
    assert simplified_out is not None
    assert torch.allclose(original_out, simplified_out)
