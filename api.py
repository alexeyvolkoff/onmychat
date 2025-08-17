from fastapi import FastAPI, HTTPException, Header, Query
from pydantic import BaseModel
import core_service
import inspect, asyncio

app = FastAPI()

# Разрешённые методы core_service
ALLOWED_METHODS = {
    "summarize_for_memory": core_service.summarize_for_memory,
    "chat": core_service.chat,
    "generate_image": core_service.generate_image,
    "recognize_image": core_service.recognize_image,
    "memorize": core_service.memorize,
    "import": core_service.import_doc,
}

class MethodCallInput(BaseModel):
    args: dict = {}

# === Авторизация ===
def get_user_context(omd_key: str = Header(..., alias="X-OMD-Key")):
    if not omd_key:
        raise HTTPException(status_code=401, detail="Missing X-OMD-Key")
    return {"omd_key": omd_key}

@app.get("/methods")
async def methods_endpoint():
    return {"methods": list(ALLOWED_METHODS.keys()) + ["history", "memory"]}

@app.post("/methods/{name}")
async def methods_method_endpoint(
    name: str,
    input: MethodCallInput,
    ctx: dict = Depends(get_user_context)
):
    if name not in ALLOWED_METHODS:
        raise HTTPException(status_code=400, detail=f"Method {name} not allowed")

    method = ALLOWED_METHODS[name]

    try:
        args = {**input.args, **ctx}  # пробрасываем omd_key в core
        if inspect.iscoroutinefunction(method):
            result = await method(**args)
        else:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: method(**args))
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# === Автогенерация алиасов ===
def create_alias(name: str, method):
    @app.post(f"/{name}")
    async def alias_endpoint(
        input: MethodCallInput,
        ctx: dict = Depends(get_user_context),
        _name=name
    ):
        return await methods_method_endpoint(_name, input, ctx)

for method_name, method_func in ALLOWED_METHODS.items():
    create_alias(method_name, method_func)

# === История ===
@app.get("/history")
async def history_endpoint(
    chat: str = Query("telegram"),
    ctx: dict = Depends(get_user_context)
):
    try:
        result = await core_service.history(chat=chat, **ctx)
        return {"history": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# === Память (jsonl) ===
@app.get("/memory")
async def memory_endpoint(ctx: dict = Depends(get_user_context)):
    try:
        jsonl = await core_service.dump_memory_jsonl(**ctx)
        return {"memory": jsonl}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
