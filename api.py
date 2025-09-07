from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import core_service
import user_context
import dialog_history
import memory_index
import logging
import json
from utils import get_image_from_source 

logging.basicConfig(level=logging.INFO)

app = FastAPI()


origins = [
    "http://localhost:8080",  
    "http://localhost:8081",  
    "*",  # Caution: Allow all origins (not recommended for production)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

class MemoryUpdate(BaseModel):
    text: str
    collection: str = "user"
    relevance: str = "contextual"
    document_id: str | None = None


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

@app.get("/assistant")
async def assistant_info(omd_key: str):
    ctx = get_ctx(omd_key)
    try:
        avatarPath = core_service.get_assistant_avatar_path(ctx.user_id)
        assistant = {
            "avatarPath": avatarPath,
            "name": ctx.settings.get("assistant_name", user_context.DEFAULT_ASSISTANT_NAME),
            "title": ctx.settings.get("assistant_title", user_context.DEFAULT_ASSISTANT_TITLE),
        }
        return assistant

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/history")
async def history_endpoint(omd_key: str, chat: str = "default"):
    ctx = get_ctx(omd_key)
    try:
        history = dialog_history.load_history(ctx, chat=chat)
        return {"chat": chat, "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory")
async def memory_endpoint(omd_key: str, collection: str = "user"):
    ctx = get_ctx(omd_key)
    try:
        memories = memory_index.load_memories(ctx, collection=collection)
        return {"collection": collection, "memories": memories}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/memory/{mem_id}")
async def update_memory(mem_id: str, data: MemoryUpdate):
    ctx = get_ctx(data.omd_key)  # сюда нужно будет пробросить omd_key
    try:
        updated_id = memory_index.update_memory_card(
            ctx=ctx,
            text=data.text,
            collection=data.collection,
            relevance=data.relevance,
            document_id=data.document_id,
            mem_id=mem_id
        )
        if not updated_id:
            raise HTTPException(status_code=404, detail="Memory not found")
        return {"status": "ok", "memory_id": updated_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memory/{mem_id}")
async def delete_memory(omd_key: str, mem_id: str, collection: str = "user"):
    ctx = get_ctx(omd_key)
    try:
        success = memory_index.delete_memory_card(ctx, mem_id, collection)
        if not success:
            raise HTTPException(status_code=404, detail="Memory not found")
        return {"status": "deleted", "memory_id": mem_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/{collection}/{mem_id}")
async def get_memory(omd_key: str, collection: str, mem_id: str):
    ctx = get_ctx(omd_key)
    try:
        memories = memory_index.load_memories(ctx, collection)
        for m in memories:
            if m["memory_id"] == mem_id:
                return m
        raise HTTPException(status_code=404, detail="Memory not found")
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

@app.get("/chat/stream")
async def chat_stream(omd_key: str, prompt: str, chat: str = "default"):
    ctx = get_ctx(omd_key)

    async def event_generator():
        # defaults
        intent = "chat"
        skip_history = False
        is_rag = False
        b64_image = None
        # check intent

        if prompt.startswith("/show"):
            intent = "show"
        elif prompt.startswith("/view") or prompt.startswith("/imagine"): 
            intent = "view"   
        elif prompt.startswith("/import") or prompt.startswith("/learn"):  
            intent = "import"   
        elif prompt.startswith("/think") or prompt.startswith("/explain"):  
            intent = "explain"   
        else:    
            intent = await core_service.classify_user_intent(ctx, prompt)

        logging.info(f"Intent detected: {intent}")
        if intent == "show":
            # 1️⃣ статус
            yield f"data: {json.dumps({'status': 'generating'})}\n\n"

            # 2️⃣ картинка
            img_prompt = await core_service.generate_character_image_prompt(ctx, prompt, chat)
            logging.info(f"Generating image for prompt {prompt}")

            path = await core_service.generate_character_image(ctx, img_prompt, chat)
            yield f"data: {json.dumps({'image':{'prompt': img_prompt, 'path': path}})}\n\n"

            #Set specific instructions
            skip_history = True
            request = (
                "Continue conversation describing how you feel in this scene: {}\n\n"
                "*'Me' refers to you in the provided scene description*"
            ).format(img_prompt)
            instruction = (
                "Recognize and describe the provided image or scene description."
            )
        elif intent == "view":
            # 1️⃣ статус
            yield f"data: {json.dumps({'status': 'generating'})}\n\n"

            # 2️⃣ картинка
            img_prompt = await core_service.generate_general_image_prompt(ctx, prompt, chat)
            logging.info(f"Generating image for prompt {prompt}")

            path = await core_service.generate_general_image(ctx, img_prompt, chat)
            yield f"data: {json.dumps({'image':{'prompt': img_prompt, 'path': path}})}\n\n"

            #Set specific instructions
            skip_history = True
            request = (
                "Continue conversation describing this scene: {}\n\n"
                "*'Me', if present, refers to you in the provided scene description*"
            ).format(img_prompt)
            instruction = (
                "Recognize and describe the provided image or scene description."
            )
        elif intent == "explain":    
            yield f"data: {json.dumps({'status': 'thinking'})}\n\n"
    
            instruction=(
                "If Known facts are provided and they are relevant to user's query, you must strictly base your response only on them. "
                "Do not invent or speculate. If the facts are insufficient to fully answer, clearly separate what is factual from what is uncertain, and explicitly state the limitations."
                "If no relevant Known facts are provided, respond freely as a helpful conversational assistant."
            )
            request = prompt
            is_rag = True
        elif intent.startswith("recognize"): 
            yield f"data: {json.dumps({'status': 'recognizing'})}\n\n"

            img_source = None
            if ":" in intent:
                img_source = intent.split(":", 1)[1]
        
            b64_image = await get_image_from_source(ctx, img_source)    

            instruction = (
                "Recognize the image according to context."
            )
            request = prompt
        elif intent.startswith("import"):
            yield f"data: {json.dumps({'status': 'importing'})}\n\n"
            doc_source = None
            card = {}
            if ":" in intent:
                doc_source = intent.split(":", 1)[1]
            if doc_source:
                card = await core_service.import_doc(ctx, doc_source)   
            new_knowledge = ""     
            if card : 
                new_knowledge = card.get("text")
            logging.info(f"*New knowledge:*\n{new_knowledge}")
            instruction=(
                f"Base your answer on *New knowledge* ONLY, if present.\n\n*New knowledge:*\n{new_knowledge}"
            )
            request = prompt
            yield f"data: {json.dumps({'status': 'thinking'})}\n\n"

        # 3️⃣ основной стрим чата
        else:
            skip_history = False
            request = prompt
            instruction = (
                "Respond to user. If user question relates to *Known facts*, be extremely accurate, do not guess."
            )
        gen = await core_service.perform_prompt(
            ctx,
            instruction=instruction,
            message=request,
            chat=chat,
            stream=True,
            skip_history=skip_history,
            is_rag = is_rag,
            b64_image=b64_image
        )
        async for chunk in gen:
            yield f"data: {json.dumps(chunk)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")




@app.post("/import")
async def import_endpoint(data: ImportInput):
    ctx = get_ctx(data.omd_key)
    try:
        card = await core_service.import_doc(ctx, data.url_or_path, data.collection)
        return {
           "status": "ok",
           "card": card
        }
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
