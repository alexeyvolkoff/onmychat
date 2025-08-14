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
import core_service as core
from utils import resize_and_base64encode, format_response_for_markdown_v2

TOKEN = SETTINGS["TELEGRAM_BOT_TOKEN"]

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)


def _ctx(user_id):
    return type("Ctx", (), {"user_id": user_id})


# ==== Commands ====

async def bind(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /bind <account_id>")
        return
    ctx = _ctx(update.effective_user.id)
    core.bind_account(ctx, context.args[0])
    await update.message.reply_text("Account bound.")

async def unbind(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ctx = _ctx(update.effective_user.id)
    core.unbind_account(ctx)
    await update.message.reply_text("Account unbound.")

async def set_system_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ctx = _ctx(update.effective_user.id)
    prompt_text = " ".join(context.args).strip()
    if not prompt_text:
        await update.message.reply_text(
            "❗ Please tell me how should I behave.\n"
            "`/system You are my companion...`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return
    settings = core.load_user_settings(ctx)
    settings["system_prompt"] = prompt_text
    core.save_user_settings(ctx, settings)
    await update.message.reply_text("✅ Okay, deal!")

async def import_doc(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /import_doc <url>")
        return
    ctx = _ctx(update.effective_user.id)
    result = await core.import_doc(ctx, context.args[0])
    await update.message.reply_text(f"Document imported: {result}")

async def remember(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /remember <text>")
        return
    ctx = _ctx(update.effective_user.id)
    result = core.memorize(ctx, " ".join(context.args))
    await update.message.reply_text(f"Memorized: {result}")

# ==== Image recognition helper ====

async def recognize_image_request(ctx, img_source: str, omd_key: str, update: Update):
    photos = update.message.photo
    message = update.message.text or "What is on this photo?"
    b64_image = None

    if photos:
        photo_file = await photos[-1].get_file()
        user_folder = f"user_data/files-{ctx.user_id}"
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
                        return "Failed to get image from On My Disk."

    if not b64_image:
        return "Image not found or could not be processed."

    recognition_prompt = re.sub(
        r"(https?://\S+)|(\S*\.(jpg|jpeg|png|gif|webp))",
        "",
        message,
        flags=re.IGNORECASE
    ).strip()

    return await core.recognize_image(
        ctx=ctx,
        instruction="Recognize the image.",
        prompt=recognition_prompt,
        img=b64_image
    )

# ==== Core Handlers ====

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ctx = _ctx(update.effective_user.id)
    settings = core.load_user_settings(ctx)
    text = update.message.text

    intent = await core.classify_user_intent(text)
    logging.info(f"Intent: {intent}")

    if intent == "show":
        path = await core.generate_character_image(ctx, text)
        await update.message.reply_photo(open(path, "rb"))

    elif intent == "view":
        path = await core.generate_general_image(ctx, text)
        await update.message.reply_photo(open(path, "rb"))

    elif intent == "explain":
        result = await core.perform_prompt(ctx, settings, "", text, is_rag=True)
        await update.message.reply_text(
            format_response_for_markdown_v2(result),
            parse_mode="MarkdownV2"
        )

    elif intent.startswith("recognize") or update.message.photo:
        img_source = None
        if ":" in intent:
            img_source = intent.split(":", 1)[1]
        result = await recognize_image_request(ctx, img_source, settings.get("omd_key"), update)
        await update.message.reply_text(format_response_for_markdown_v2(result), parse_mode="MarkdownV2")

    else:
        result = await core.perform_prompt(ctx, settings, "", text)
        await update.message.reply_text(format_response_for_markdown_v2(result), parse_mode="MarkdownV2")


async def ask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ctx = _ctx(update.effective_user.id)
    query = " ".join(context.args) if context.args else update.message.text.strip()
    settings = core.load_user_settings(ctx)
    result = await core.perform_prompt(
        ctx, settings,
        "Use the *Known facts* provided above. If no info, say you do not know.",
        query, is_rag=True
    )
    await update.message.reply_text(format_response_for_markdown_v2(result), parse_mode="MarkdownV2")

async def nsfw_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ctx = _ctx(update.effective_user.id)
    args = context.args
    if not args or args[0].lower() not in ['on', 'off']:
        await update.message.reply_text("❗ Usage: /nsfw on | off")
        return
    nsfw_enabled = args[0].lower() == 'on'
    settings = core.load_user_settings(ctx)
    settings["nsfw"] = nsfw_enabled
    core.save_user_settings(ctx, settings)
    status = "enabled 🔞" if nsfw_enabled else "disabled ✅"
    await update.message.reply_text(f"NSFW mode {status}")

async def set_style(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ctx = _ctx(update.effective_user.id)
    if not context.args:
        await update.message.reply_text("❗ Choose style: realistic, dream, tooned.")
        return
    chosen_style = context.args[0].lower()
    if chosen_style not in core.STYLE_MODELS:
        await update.message.reply_text(f"❗ Unknown style. Supported: {', '.join(core.STYLE_MODELS.keys())}")
        return
    settings = core.load_user_settings(ctx)
    settings["style"] = chosen_style
    core.save_user_settings(ctx, settings)
    await update.message.reply_text(f"✅ Style set to *{chosen_style}*", parse_mode="Markdown")

# ==== Main ====

def main():
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("bind", bind))
    app.add_handler(CommandHandler("unbind", unbind))
    app.add_handler(CommandHandler("system", set_system_prompt))
    app.add_handler(CommandHandler("import_doc", import_doc))
    app.add_handler(CommandHandler("remember", remember))
    app.add_handler(CommandHandler("ask", ask))
    app.add_handler(CommandHandler("recognize", handle_text))
    app.add_handler(CommandHandler("nsfw", nsfw_command))
    app.add_handler(CommandHandler("style", set_style))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.PHOTO, handle_text))

    app.run_polling()

if __name__ == "__main__":
    main()
