import logging
import os
from telegram import Update
from telegram.constants import ParseMode, ChatAction
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
)
from config import SETTINGS
from config import USER_DATA_DIR
import core_service as core
from utils import resize_and_base64encode, get_image_from_source, format_response_for_markdown_v2, escape_markdown_v2
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
    if card:
        await update.message.reply_text(f"✅ Document imported: `{card["id"]}`", parse_mode="Markdown")
    else:
        await update.message.reply_text("❗Failed to import")    

async def remember(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /remember <text>")
        return
    ctx = get_context(update.effective_user.id)
    result = core.memorize(ctx, " ".join(context.args))
    await update.message.reply_text(f"✅ Memorized: `{result}`", parse_mode="Markdown")

# ==== Image recognition helper ====

async def get_image_from_request(ctx, img_source: str,  update: Update):
    photos = update.message.photo
    omd_key = ctx.settings.get("omd_key")
    b64_image = None

    if photos:
        photo_file = await photos[-1].get_file()
        user_folder = f"{USER_DATA_DIR}/{ctx.user_id}/files"
        os.makedirs(user_folder, exist_ok=True)
        file_path = os.path.join(user_folder, os.path.basename(photo_file.file_path))
        await photo_file.download_to_drive(file_path)
        b64_image = resize_and_base64encode(file_path)
    elif img_source:
        b64_image = await get_image_from_source(ctc, img_source)
    return b64_image                

    

# ==== Core Handlers ====

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE, command = None):
    ctx = get_context(update.effective_user.id)
    text = update.message.text or ""
    intent = "chat"
    if not text and update.message.photo:
        intent = "recognize"
    else:    
        raw_intent = command if command else await core.classify_user_intent(ctx, text)
        lines = raw_intent.strip().split("\n", 1)
        intent = lines[0].strip()
        if len(lines) > 1:
            logging.info(f"Intent explanation: {lines[1].strip()}")

    logging.info(f"Intent: {intent}")

    chat_id = update.effective_chat.id
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
 
    if intent == "show":
        prompt = await core.generate_character_image_prompt(ctx, text, "telegram")
        logging.info(f"Generating character prompt {intent} ")
        await update.message.chat.send_action(action=ChatAction.TYPING)
        logging.info(f"Generating image for prompt {prompt} ")
        # Отправляем фото
        storage_path = await core.generate_character_image(ctx, prompt, "telegram")
        path = f"{core.COMFY_OUTPUT_DIR}/{storage_path}"
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
               "Continue conversation describing how you feel in this scene: {}" 
            ).format(prompt)
            response = await core.perform_prompt(
                ctx,
                instruction=(
                    "Recognize and describe the provided image or scene description."
                ),
                message=recognition_prompt,
                chat="telegram", 
                skip_history=True
            )
            explained = response.get("content") or "✅ done" 
            result = format_response_for_markdown_v2(explained)
            await update.message.reply_text(result, parse_mode="MarkdownV2")
            logging.info(f"Explained photo: {explained}")

    elif intent == "view":
        prompt = await core.generate_general_image_prompt(ctx, text, "telegram")
        await update.message.chat.send_action(action=ChatAction.TYPING)
        storage_path = await core.generate_general_image(ctx, prompt, "telegram")
        path = f"{core.COMFY_OUTPUT_DIR}/{storage_path}"
        # Отправляем фото
        with open(path, "rb") as f:
            await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.UPLOAD_PHOTO)
            caption = escape_markdown_v2(prompt)
            if len(caption) > MAX_CAPTION_LEN:
               caption = caption[:MAX_CAPTION_LEN - 1] + "…"
            await update.message.reply_photo(photo=f, caption=caption, parse_mode="MarkdownV2")
            await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            # Объясняем картинку
            recognition_prompt = (
               "Continue conversation according the view or the scenary: {}"
            ).format(prompt)
            response = await core.perform_prompt(
                ctx,
                instruction=(
                    "Describe the provided image or scenary."
                ),
                message=recognition_prompt,
                #b64_image=b64_image,
                chat="telegram",
                skip_history=False
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
        links = []
        doc_ids = response.get("sources")
        if doc_ids:
            for doc in doc_ids:
                label = doc
                # Формируем ссылку без экранирования, она будет безопасно обработана позже
                links.append(f"• [{label}]({doc})")
        if links:
            result += "\n\n📎 *Sources:*\n" + '\n'.join(links)
        await update.message.reply_text(
            format_response_for_markdown_v2(result),
            parse_mode="MarkdownV2"
        )

    elif intent.startswith("recognize") or update.message.photo:
        img_source = None
        if ":" in intent:
            img_source = intent.split(":", 1)[1]
        
        recognition_prompt = (
            "Recognize the image according to context."
        )

        response = await core.perform_prompt(
            ctx,
            instruction=(
                "Recognize and describe the provided images."
            ),
            message=recognition_prompt,
            img_source=img_source,
            chat="telegram"
        )
        explained = response.get("content") or "✅ done" 
        result = format_response_for_markdown_v2(explained)
        await update.message.reply_text(result, parse_mode="MarkdownV2")    

    elif intent.startswith("import"):
        doc_source = None
        card = {}
        if ":" in intent:
            doc_source = intent.split(":", 1)[1]
        if doc_source:
            card = await core.import_doc(ctx, doc_source)   
        new_knowledge = ""     
        if card : 
            new_knowledge = card.get("text")
            logging.info(f"*New knowledge:*\n{new_knowledge}")

        response = await core.perform_prompt(
            ctx,
            instruction=(
                f"Summirize the *New knowledge* ONLY, if present.\n\n*New knowledge:*\n{new_knowledge}"
            ),
            message=text,
            chat="telegram"
        )
        explained = response.get("content") or "✅ done" 
        links = []
        doc_ids = response.get("sources")
        if doc_ids:
            for doc in doc_ids:
                label = doc
                # Формируем ссылку без экранирования, она будет безопасно обработана позже
                links.append(f"• [{label}]({doc})")
        if links:
            explained += "\n\n📎 *Sources:*\n" + '\n'.join(links)

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
        links = []
        doc_ids = response.get("sources")
        if doc_ids:
            for doc in doc_ids:
                label = doc
                # Формируем ссылку без экранирования, она будет безопасно обработана позже
                links.append(f"• [{label}]({doc})")
        if links:
            result += "\n\n📎 *Sources:*\n" + '\n'.join(links)
        await update.message.reply_text(format_response_for_markdown_v2(result), parse_mode="MarkdownV2")


async def explain(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ctx = get_context(update.effective_user.id)
    args = context.args
    if not args:
        await update.message.reply_text("❗ Usage: /explain <your question>")
        return
    await handle_text(update, context, "explain")

async def show_character(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ctx = get_context(update.effective_user.id)
    args = context.args
    if not args:
        await update.message.reply_text("❗ Usage: /show <your request>")
        return
    await handle_text(update, context, "show")

async def show_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ctx = get_context(update.effective_user.id)
    args = context.args
    if not args:
        await update.message.reply_text("❗ Usage: /image <your request>")
        return
    await handle_text(update, context, "view")


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
    app.add_handler(CommandHandler("explain", explain))
    app.add_handler(CommandHandler("show", show_character))
    app.add_handler(CommandHandler("image", show_image))
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
