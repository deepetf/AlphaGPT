from __future__ import annotations

from dataclasses import dataclass

from .ops_registry import OpsRegistry


COMMUTATIVE_OPS = {"ADD", "MUL", "MIN", "MAX"}
IDEMPOTENT_OPS = {"MIN", "MAX"}
REDUNDANT_UNARY_OPS = {"ABS", "CUT_NEG"}


@dataclass(frozen=True)
class FormulaNode:
    token: str
    children: tuple["FormulaNode", ...] = ()

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


def _ops_arity_map() -> dict[str, int]:
    return {name: arity for name, _, arity in OpsRegistry.get_ops_config()}


def parse_rpn(formula: list[str]) -> FormulaNode:
    if not formula:
        raise ValueError("formula is empty")

    arity_map = _ops_arity_map()
    stack: list[FormulaNode] = []
    for token in formula:
        arity = arity_map.get(token)
        if arity is None:
            stack.append(FormulaNode(token=token))
            continue

        if len(stack) < arity:
            raise ValueError(f"invalid RPN formula, token='{token}' lacks operands")

        args = stack[-arity:]
        del stack[-arity:]
        stack.append(FormulaNode(token=token, children=tuple(args)))

    if len(stack) != 1:
        raise ValueError("invalid RPN formula, stack does not reduce to one result")
    return stack[0]


def _canonical_tuple(node: FormulaNode) -> tuple:
    return (node.token, tuple(_canonical_tuple(child) for child in node.children))


def _sort_children(children: tuple[FormulaNode, ...]) -> tuple[FormulaNode, ...]:
    return tuple(sorted(children, key=_canonical_tuple))


def _flatten_same_op(node: FormulaNode) -> list[FormulaNode]:
    flat: list[FormulaNode] = []
    for child in node.children:
        if child.token == node.token:
            flat.extend(_flatten_same_op(child))
        else:
            flat.append(child)
    return flat


def _build_balanced_binary(token: str, children: list[FormulaNode]) -> FormulaNode:
    if not children:
        raise ValueError(f"cannot build binary node '{token}' with empty children")
    if len(children) == 1:
        return children[0]

    current = children[0]
    for child in children[1:]:
        current = FormulaNode(token=token, children=(current, child))
    return current


def _apply_absorption(token: str, left: FormulaNode, right: FormulaNode) -> FormulaNode | None:
    if token == "MAX":
        if right.token == "MIN" and any(child == left for child in _flatten_same_op(right)):
            return left
        if left.token == "MIN" and any(child == right for child in _flatten_same_op(left)):
            return right
    elif token == "MIN":
        if right.token == "MAX" and any(child == left for child in _flatten_same_op(right)):
            return left
        if left.token == "MAX" and any(child == right for child in _flatten_same_op(left)):
            return right
    return None


def simplify_ast(node: FormulaNode) -> FormulaNode:
    if node.is_leaf:
        return node

    simplified_children = tuple(simplify_ast(child) for child in node.children)
    node = FormulaNode(token=node.token, children=simplified_children)

    if node.token in COMMUTATIVE_OPS:
        node = FormulaNode(token=node.token, children=_sort_children(node.children))

    if node.token in REDUNDANT_UNARY_OPS and len(node.children) == 1:
        child = node.children[0]
        if child.token == node.token:
            return child

    if node.token in IDEMPOTENT_OPS and len(node.children) == 2:
        left, right = node.children
        if left == right:
            return left

        absorbed = _apply_absorption(node.token, left, right)
        if absorbed is not None:
            return absorbed

        flat_children = _flatten_same_op(node)
        dedup_children: list[FormulaNode] = []
        seen = set()
        for child in flat_children:
            key = _canonical_tuple(child)
            if key in seen:
                continue
            seen.add(key)
            dedup_children.append(child)
        dedup_children = list(_sort_children(tuple(dedup_children)))
        if len(dedup_children) == 1:
            return dedup_children[0]
        return _build_balanced_binary(node.token, dedup_children)

    return node


def canonicalize_ast(node: FormulaNode) -> FormulaNode:
    if node.is_leaf:
        return node

    children = tuple(canonicalize_ast(child) for child in node.children)
    if node.token in COMMUTATIVE_OPS:
        children = _sort_children(children)
    return FormulaNode(token=node.token, children=children)


def ast_to_rpn(node: FormulaNode) -> list[str]:
    if node.is_leaf:
        return [node.token]

    out: list[str] = []
    for child in node.children:
        out.extend(ast_to_rpn(child))
    out.append(node.token)
    return out


def simplify_formula(formula: list[str]) -> list[str]:
    try:
        root = parse_rpn(formula)
    except Exception:
        return list(formula)

    simplified = simplify_ast(root)
    canonical = canonicalize_ast(simplified)
    return ast_to_rpn(canonical)


def formula_to_canonical_key(formula: list[str]) -> tuple[str, ...]:
    return tuple(simplify_formula(formula))
