import random

tool = {
    "type": "function",
    "function": {
        "name": "roll_dice",
        "description": "Roll one or more dice with the specified number of sides.",
        "parameters": {
            "type": "object",
            "properties": {
                "count": {"type": "integer", "description": "Number of dice to roll.", "default": 1},
                "sides": {"type": "integer", "description": "Number of sides per die.", "default": 20},
            },
        }
    }
}


def execute(arguments):
    count = arguments.get("count", 1)
    sides = arguments.get("sides", 20)
    rolls = [random.randint(1, sides) for _ in range(count)]
    return {"rolls": rolls, "total": sum(rolls)}
