import uuid
import time

USERS = {"wsm": "666666"}
_tokens: dict[str, dict] = {}


def login(username: str, password: str) -> dict:
    if username not in USERS or USERS[username] != password:
        return {"success": False, "message": "用户名或密码错误"}
    token = str(uuid.uuid4())
    _tokens[token] = {"username": username, "created": time.time()}
    return {"success": True, "token": token, "username": username}


def validate_token(token: str) -> str | None:
    data = _tokens.get(token)
    if not data:
        return None
    if time.time() - data["created"] > 86400:
        _tokens.pop(token, None)
        return None
    return data["username"]


def logout(token: str):
    _tokens.pop(token, None)
