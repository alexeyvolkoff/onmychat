from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from pydantic import BaseModel
import core_service
import user_context
import dialog_history
import memory_index
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    user_context.load_bindings()
    logging.info("[api] Bindings loaded")

# ==== Модели ввода ====

class ChatInput(BaseModel):
    omd_key: str
    prompt: str
    chat: str = "default"

class ImportInput(BaseModel):
    omd_key: str
    url_or_path: str
    collection: str = "user"

class MemorizeInput(BaseModel):
    omd_key: str
    text: str

class RecognizeInput(BaseModel):
    omd_key: str
    prompt: str = ""
    chat: str = "default"

class GenerateInput(BaseModel):
    omd_key: str
    prompt: str
    chat: str = "default"


# ==== Хелпер ====

def get_ctx(omd_key: str):
    ctx = user_context.get_context_by_account(omd_key)
    if ctx.type == "temp":
        raise HTTPException(status_code=401, detail="Invalid or unbound OMD key")
    return ctx


# ==== Эндпоинты ====


@app.get("/history")
async def history_endpoint(omd_key: str, chat: str = "default"):
    ctx = get_ctx(omd_key)
    try:
        history = dialog_history.load_history(ctx.user_id, chat=chat)
        return {"chat": chat, "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory")
async def memory_endpoint(omd_key: str, collection: str = "user"):
    ctx = get_ctx(omd_key)
    try:
        memories = memory_index.load_memories(ctx.user_id, collection=collection)
        return {"collection": collection, "memories": memories}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chats")
async def chats_endpoint(omd_key: str):
    ctx = get_ctx(omd_key)
    try:
        from dialog_history import load_chats_index
        chats = load_chats_index(ctx.user_id)
        return chats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_endpoint(data: ChatInput):
    ctx = get_ctx(data.omd_key)
    try:
        instruction=(
            "Respond to user. If user question relates to *Known facts*, be extreamly accurate, do not guess."
        )
        response = await core_service.perform_prompt(
            ctx,
            instruction=instruction,
            message=data.prompt,
            chat=data.chat,
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/import")
async def import_endpoint(data: ImportInput):
    ctx = get_ctx(data.omd_key)
    try:
        await core_service.import_doc(ctx, data.url_or_path, data.collection)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memorize")
async def memorize_endpoint(data: MemorizeInput):
    ctx = get_ctx(data.omd_key)
    try:
        core_service.memorize(ctx, data.text)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recognize")
async def recognize_endpoint(
    omd_key: str = Form(...),
    chat: str = Form("default"),
    prompt: str = Form(""),
    file: UploadFile = File(...)
):
    ctx = get_ctx(omd_key)
    try:
        img_bytes = await file.read()
        result = await core_service.recognize_image(ctx, img_bytes, prompt, chat)
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/image/character")
async def generate_character_image(data: GenerateInput):
    ctx = get_ctx(data.omd_key)
    try:
        result = await core_service.generate_character_image(ctx, data.prompt)
        return {"image": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/image/general")
async def generate_general_image(data: GenerateInput):
    ctx = get_ctx(data.omd_key)
    try:
        result = await core_service.generate_general_image(ctx, data.prompt)
        return {"image": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/prompt/character")
async def generate_character_image_prompt(data: GenerateInput):
    ctx = get_ctx(data.omd_key)
    try:
        result = await core_service.generate_character_image_prompt(ctx, data.prompt, data.chat)
        return {"prompt": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/prompt/general")
async def generate_general_image_prompt(data: GenerateInput):
    ctx = get_ctx(data.omd_key)
    try:
        result = await core_service.generate_general_image_prompt(ctx, data.prompt, data.chat)
        return {"prompt": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
