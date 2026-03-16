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


def ast_to_expression(node: FormulaNode) -> str:
    if node.is_leaf:
        return node.token

    rendered_children = [ast_to_expression(child) for child in node.children]
    if len(rendered_children) == 1:
        return f"{node.token}({rendered_children[0]})"
    return f"{node.token}(" + ", ".join(rendered_children) + ")"


def expand_formula(formula: list[str]) -> str:
    root = parse_rpn(formula)
    return ast_to_expression(root)


def ast_to_semantic_expression(node: FormulaNode) -> str:
    if node.is_leaf:
        return node.token

    rendered_children = [ast_to_semantic_expression(child) for child in node.children]
    token = node.token

    if len(rendered_children) == 1:
        child = rendered_children[0]
        if token == "NEG":
            return f"(-{child})"
        if token == "ABS":
            return f"abs({child})"
        return f"{token.lower()}({child})"

    if len(rendered_children) == 2:
        left, right = rendered_children
        if token == "ADD":
            return f"({left} + {right})"
        if token == "SUB":
            return f"({left} - {right})"
        if token == "MUL":
            return f"({left} * {right})"
        if token == "DIV":
            return f"safe_div({left}, {right})"
        if token == "MIN":
            return f"min({left}, {right})"
        if token == "MAX":
            return f"max({left}, {right})"
        if token == "IF_POS":
            return f"({left} if {left} > 0 else {right})"

    return f"{token.lower()}(" + ", ".join(rendered_children) + ")"


def expand_formula_semantic(formula: list[str]) -> str:
    root = parse_rpn(formula)
    return ast_to_semantic_expression(root)


def _is_nonnegative_expr(node: FormulaNode) -> bool:
    if node.is_leaf:
        return False
    if node.token in {"ABS", "SQRT", "CUT_NEG", "TS_STD5", "TS_STD20", "TS_STD60"}:
        return True
    if node.token in {"MIN", "MAX"}:
        return all(_is_nonnegative_expr(child) for child in node.children)
    return False


def _collect_structure_hints(node: FormulaNode, hints: list[str]) -> None:
    for child in node.children:
        _collect_structure_hints(child, hints)

    if node.token == "ABS" and len(node.children) == 1 and node.children[0].token == "NEG":
        inner = ast_to_expression(node.children[0].children[0]) if node.children[0].children else ast_to_expression(node.children[0])
        hints.append(f"ABS(NEG({inner})) 在代数上可化简为 ABS({inner})。")

    if node.token == "IF_POS" and len(node.children) == 2:
        condition = node.children[0]
        condition_expr = ast_to_expression(condition)
        if _is_nonnegative_expr(condition):
            hints.append(
                f"IF_POS({condition_expr}, ...) 的条件输入按算子定义为非负量；其 else 分支通常只在 {condition_expr} = 0 时触发。"
            )

    if node.token == "MIN" and len(node.children) == 2:
        left, right = node.children
        if left.is_leaf and not right.is_leaf:
            cap_expr = left.token
        elif right.is_leaf and not left.is_leaf:
            cap_expr = right.token
        else:
            cap_expr = ""
        if cap_expr:
            hints.append(f"MIN(..., {cap_expr}) 会对输出施加不高于 {cap_expr} 的上界截断。")


def collect_structure_hints(formula: list[str]) -> list[str]:
    root = parse_rpn(formula)
    hints: list[str] = []
    _collect_structure_hints(root, hints)
    deduped: list[str] = []
    for hint in hints:
        if hint not in deduped:
            deduped.append(hint)
    return deduped


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
