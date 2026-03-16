import ast
import operator

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    elif isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        left = _eval(node.left)
        right = _eval(node.right)
        if isinstance(node.op, ast.Pow) and isinstance(right, (int, float)) and abs(right) > 10000:
            raise ValueError("Exponent too large (max 10000)")
        return OPERATORS[type(node.op)](left, right)
    elif isinstance(node, ast.UnaryOp) and type(node.op) in OPERATORS:
        return OPERATORS[type(node.op)](_eval(node.operand))
    raise ValueError(f"Unsupported expression")


tool = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Evaluate a math expression. Supports +, -, *, /, **, %.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "The math expression to evaluate (e.g. '2 * (3 + 4)')."},
            },
            "required": ["expression"]
        }
    }
}


def execute(arguments):
    expr = arguments.get("expression", "")
    try:
        tree = ast.parse(expr, mode='eval')
        result = _eval(tree.body)
        return {"expression": expr, "result": result}
    except Exception as e:
        return {"error": str(e)}
