from datetime import datetime

tool = {
    "type": "function",
    "function": {
        "name": "get_datetime",
        "description": "Get the current date and time.",
        "parameters": {
            "type": "object",
            "properties": {},
        }
    }
}


def execute(arguments):
    now = datetime.now()
    return {"date": now.strftime("%Y-%m-%d"), "time": now.strftime("%I:%M %p")}
