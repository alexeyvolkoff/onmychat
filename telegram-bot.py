import logging
import os
import re
import base64
import aiohttp
from io import BytesIO
from telegram import Update
from telegram.constants import ParseMode, ChatAction
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
)
from config import SETTINGS
from config import USER_DATA_DIR
import core_service as core
from utils import resize_and_base64encode, format_response_for_markdown_v2, escape_markdown_v2
from user_context import (load_bindings, get_context, save_user_settings)

TOKEN = SETTINGS["TELEGRAM_BOT_TOKEN"]
MAX_CAPTION_LEN = 1024 # байт

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

# ==== Commands ====

async def bind(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /bind <account_id>")
        return
    ctx = get_context(update.effective_user.id)
    ctx = core.bind_account(ctx, context.args[0])
    await update.message.reply_text(f"✅ Account bound: `{ctx.user_id}`", parse_mode="Markdown")

async def set_system_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ctx = get_context(update.effective_user.id)
    prompt_text = " ".join(context.args).strip()
    if not prompt_text:
        await update.message.reply_text(
            "❗ Please tell me how should I behave.\n"
            "`/system You are my companion...`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return
    ctx.settings["system_prompt"] = prompt_text
    save_user_settings(ctx)
    await update.message.reply_text("✅ Okay, deal!")

async def import_doc(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /import <url> [kb_id]")
        return
    ctx = get_context(update.effective_user.id)
    collection = context.args[1].strip().lower() if len(context.args) > 1 else "user"
    card = await core.import_doc(ctx, context.args[0], collection)
    await update.message.reply_text(f"✅ Document imported: `{card["id"]}`", parse_mode="Markdown")

async def remember(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /remember <text>")
        return
    ctx = get_context(update.effective_user.id)
    result = core.memorize(ctx, " ".join(context.args))
    await update.message.reply_text(f"✅ Memorized: `{result}`", parse_mode="Markdown")

# ==== Image recognition helper ====

async def get_image_from_request(ctx, img_source: str, omd_key: str, update: Update):
    photos = update.message.photo
    b64_image = None

    if photos:
        photo_file = await photos[-1].get_file()
        user_folder = f"{USER_DATA_DIR}/{ctx.user_id}/files"
        os.makedirs(user_folder, exist_ok=True)
        file_path = os.path.join(user_folder, os.path.basename(photo_file.file_path))
        await photo_file.download_to_drive(file_path)
        b64_image = resize_and_base64encode(file_path)
    elif img_source:
        if re.match(r"^https?://", img_source):
            async with aiohttp.ClientSession() as session:
                async with session.get(img_source) as resp:
                    data = await resp.read()
                    b64_image = base64.b64encode(data).decode("utf-8")
        elif os.path.exists(img_source):
            b64_image = resize_and_base64encode(img_source)
        elif img_source.startswith("/") and omd_key:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"https://onmydisk.net{img_source}?resize=true&width=512&height=512",
                    headers={"Authorization": f"token:{omd_key}"}
                ) as resp:
                    if resp.status == 200:
                        data = await resp.read()
                        b64_image = base64.b64encode(data).decode("utf-8")
                    else:
                        logging.error(f"OMD access failed: {resp.status}")
                        return None
    return b64_image                

    

# ==== Core Handlers ====

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ctx = get_context(update.effective_user.id)
    text = update.message.text or ""
    intent = "chat"
    if not text and update.message.photo:
        intent = "recognize"
    else:    
        intent = await core.classify_user_intent(text)
    logging.info(f"Intent: {intent}")
    chat_id = update.effective_chat.id
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    logging.info(f"Generating characted prompt {intent} ")
 
    if intent == "show" or intent == "'show'":
        logging.info(f"Generating characted prompt ")
        prompt = await core.generate_character_image_prompt(ctx, text, "telegram")
        await update.message.chat.send_action(action=ChatAction.TYPING)
        # Отправляем фото
        path = await core.generate_character_image(ctx, prompt)
        b64_image = resize_and_base64encode(path)
        with open(path, "rb") as f:
            # Отправляем фото
            await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.UPLOAD_PHOTO)
            caption = escape_markdown_v2(prompt)
            if len(caption) > MAX_CAPTION_LEN:
               caption = caption[:MAX_CAPTION_LEN - 1] + "…"
            await update.message.reply_photo(photo=f, caption=caption, parse_mode="MarkdownV2")
            # Объясняем картинку
            await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            recognition_prompt = (
               "describe the scene in first person and express how you feel in it" 
            )
            response = await core.perform_prompt(
                ctx,
                instruction=(
                    "Recognize and describe the provided images. "
                    "If an image is provided, follow the user's request precisely, without adding unrelated details or commentary."
                ),
                message=recognition_prompt,
                b64_image=b64_image,
                chat="telegram"
            )
            explained = response.get("content") or "✅ done" 
            result = format_response_for_markdown_v2(explained)
            await update.message.reply_text(result, parse_mode="MarkdownV2")
            logging.info(f"Explained photo: {explained}")

    elif intent == "view":
        prompt = await core.generate_general_image_prompt(ctx, text, "telegram")
        await update.message.chat.send_action(action=ChatAction.TYPING)
        path = await core.generate_general_image(ctx, prompt)
        b64_image = resize_and_base64encode(path)
        with open(path, "rb") as f:
            # Отправляем фото
            await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.UPLOAD_PHOTO)
            caption = escape_markdown_v2(prompt)
            if len(caption) > MAX_CAPTION_LEN:
               caption = caption[:MAX_CAPTION_LEN - 1] + "…"
            await update.message.reply_photo(photo=f, caption=caption, parse_mode="MarkdownV2")
            await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            # Объясняем картинку
            recognition_prompt = (
               "view: '{}' - summarize briefly, what do you see on this photo and how do you feel about it"
            ).format(text)
            response = await core.perform_prompt(
                ctx,
                instruction=(
                    "Recognize and describe the provided images. "
                    "If an image is provided, follow the user's request precisely, without adding unrelated details or commentary."
                ),
                message=recognition_prompt,
                b64_image=b64_image,
                chat="telegram"
            )
            explained = response.get("content") or "✅ done" 
            result = format_response_for_markdown_v2(explained)
            await update.message.reply_text(result, parse_mode="MarkdownV2")
            logging.info(f"Explained photo: {explained}")


    elif intent == "explain":
        instruction=(
            "If Known facts are provided and they are relevant to user's query, you must strictly base your response only on them. "
            "Do not invent or speculate. If the facts are insufficient to fully answer, clearly separate what is factual from what is uncertain, and explicitly state the limitations."
            "If no relevant Known facts are provided, respond freely as a helpful conversational assistant."
        )
        response = await core.perform_prompt(ctx, instruction, text, is_rag=True, chat="telegram")
        result = response.get("content") or "✅ done" 
        links = response.get("sources")
        if links:
            result += "\n\n📎 *Sources:*\n" + links
        await update.message.reply_text(
            format_response_for_markdown_v2(result),
            parse_mode="MarkdownV2"
        )

    elif intent.startswith("recognize") or update.message.photo:
        img_source = None
        if ":" in intent:
            img_source = intent.split(":", 1)[1]
        
        b64_image = await get_image_from_request(ctx, img_source, ctx.settings.get("omd_key"), update)    

        recognition_prompt = (
            "Recognize the image according to context."
        )
        response = await core.perform_prompt(
            ctx,
            instruction=(
                "Recognize and describe the provided images."
            ),
            message=recognition_prompt,
            b64_image=b64_image,
            chat="telegram"
        )
        explained = response.get("content") or "✅ done" 
        result = format_response_for_markdown_v2(explained)
        await update.message.reply_text(result, parse_mode="MarkdownV2")    

    elif intent.startswith("import") or update.message.photo:
        doc_source = None
        card = {}
        if ":" in intent:
            doc_source = intent.split(":", 1)[1]
        if doc_source:
            card = await core.import_doc(ctx, doc_source)    
        if card.get("text"): 
            logging.info(f"*New knowledge:*\n{card.get("text")}")

        response = await core.perform_prompt(
            ctx,
            instruction=(
                f"Summirize the *New knowledge* ONLY, if present.\n\n*New knowledge:*\n{card.get("text")}"
            ),
            message=text,
            chat="telegram"
        )
        explained = response.get("content") or "✅ done" 
        result = format_response_for_markdown_v2(explained)
        await update.message.reply_text(result, parse_mode="MarkdownV2")    


    else:
        instruction=(
            "If Known facts are provided and they are relevant to user's query, be extreamly accurate and base your response only on them. "
            "Do not invent or speculate. If the facts are insufficient to fully answer, clearly separate what is factual from what is uncertain, "
            "and explicitly state the limitations."
            "If no relevant Known facts are provided, respond freely as a helpful conversational assistant."
        )
        response = await core.perform_prompt(ctx, instruction=instruction, message=text, is_rag=False, chat="telegram")
        result = response.get("content") or "✅ done"  
        links = response.get("sources")
        if links:
            result += "\n\n📎 *Sources:*\n" + links
        await update.message.reply_text(format_response_for_markdown_v2(result), parse_mode="MarkdownV2")


async def ask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ctx = get_context(update.effective_user.id)
    args = context.args
    if not args:
        await update.message.reply_text("❗ Usage: /ask <your questing>")
        return

    query = " ".join(args) if args else update.message.text.strip()
    chat_id = update.effective_chat.id
    await update.message.chat.send_action(chat_id=chat_id, action=ChatAction.TYPING)

    response = await core.perform_prompt(
        ctx,
        "Use the *Known facts* provided above. If no info, say you do not know.",
        query, is_rag=True, chat="telegram"
    )
    result = response.get("content") or "✅ done"  
    links = response.get("sources")
    if links:
        result += "\n\n📎 *Sources:*\n" + links

    await update.message.reply_text(format_response_for_markdown_v2(result), parse_mode="MarkdownV2")

async def nsfw_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ctx = get_context(update.effective_user.id)
    args = context.args
    if not args or args[0].lower() not in ['on', 'off']:
        await update.message.reply_text("❗ Usage: /nsfw on | off")
        return
    nsfw_enabled = args[0].lower() == 'on'
    ctx.settings["nsfw"] = nsfw_enabled
    save_user_settings(ctx)
    status = "enabled 🔞" if nsfw_enabled else "disabled ✅"
    await update.message.reply_text(f"NSFW mode {status}")

async def set_style(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ctx = get_context(update.effective_user.id)
    if not context.args:
        await update.message.reply_text("❗ Choose style: realistic, dream, tooned.")
        return
    chosen_style = context.args[0].lower()
    if chosen_style not in core.STYLE_MODELS:
        await update.message.reply_text(f"❗ Unknown style. Supported: {', '.join(core.STYLE_MODELS.keys())}")
        return
    ctx.settings["style"] = chosen_style
    save_user_settings(ctx)
    await update.message.reply_text(f"✅ Style set to *{chosen_style}*", parse_mode="Markdown")


async def get_settings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ctx = get_context(update.effective_user.id)
    system_prompt = ctx.settings.get("system_prompt", "—")
    nsfw = "enabled" if ctx.settings.get("nsfw", False) else "disabled"
    knowledge = ctx.settings.get("kb_id", "")
    style = ctx.settings.get("style", "realistic")
    storage = ctx.settings.get("storage", "")

    msg = (
        f"*User settings:*\n"
        f"• User: `{ctx.user_id}`\n"
        f"• Style: `{style}`\n"
        f"• Knowledge: `{knowledge}`\n"
        f"• Storage: `{storage}`\n"
        f"• System prompt:\n`{system_prompt}`"
    )
    if nsfw == "enabled":
        msg += f"\n• NSFW: `{nsfw}`"

    await update.message.reply_text(msg, parse_mode="Markdown")

async def useknowledge(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ctx = get_context(update.effective_user.id)
    if not context.args:
        await update.message.reply_text(
            "❗ Specify knowledge base, for example:\n`/useknowledge kb-abc123`",
            parse_mode="Markdown"
        )
        return

    kb_id = context.args[0]
    
    ctx.settings["kb_id"] = kb_id
    save_user_settings(ctx)

    await update.message.reply_text(f"✅ Switched to `{kb_id}` knowledge base.", parse_mode="Markdown")


async def setstorage(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ctx = get_context(update.effective_user.id)
    if not context.args:
        await update.message.reply_text(
            "❗ Specify setstorage path, for example:\n`/mylaptop/data/onmychat`",
            parse_mode="Markdown"
        )
        return

    storage = context.args[0]

    ctx.settings["storage"] = storage
    save_user_settings(ctx)
    await update.message.reply_text(f"✅ Storage for your data: `{storage}`", parse_mode="Markdown")
 
    #move personal data
    if ctx.settings.get("omd_key"):
        ctx = core.bind_account(ctx, ctx.settings["omd_key"])
        await update.message.reply_text(f"✅ Personal data moved to storage: `{storage}`", parse_mode="Markdown")



# ==== Main ====

def main():

    load_bindings()

    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("bind", bind))
    app.add_handler(CommandHandler("system", set_system_prompt))
    app.add_handler(CommandHandler("import", import_doc))
    app.add_handler(CommandHandler("remember", remember))
    app.add_handler(CommandHandler("ask", ask))
    app.add_handler(CommandHandler("recognize", handle_text))
    app.add_handler(CommandHandler("nsfw", nsfw_command))
    app.add_handler(CommandHandler("style", set_style))
    app.add_handler(CommandHandler("showsettings", get_settings))
    app.add_handler(CommandHandler("useknowledge", useknowledge))
    app.add_handler(CommandHandler("setstorage", setstorage))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.PHOTO, handle_text))

    app.run_polling()

if __name__ == "__main__":
    main()
