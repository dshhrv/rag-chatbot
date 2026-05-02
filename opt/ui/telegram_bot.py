import asyncio
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = ROOT / "opt" / "rag" / "src"
AGENT_PATH = SRC_PATH / "agent"

sys.path.insert(0, str(AGENT_PATH))
sys.path.insert(0, str(SRC_PATH))

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from agent_router import run_agent_timed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TOKEN = "8625198384:AAFSSRiKOXXN6FLnOxgoaealmtyEhOqVnj4"
bot = Bot(token=TOKEN)
dp = Dispatcher()

@dp.message(Command("start"))
async def start(message: types.Message):
    await message.answer("Привет! Задай вопрос по Попаткусу.")

@dp.message()
async def handle_message(message: types.Message):
    query = message.text
    user_id = message.from_user.id
    logger.info(f"{user_id}: {query[:100]}")
    await bot.send_chat_action(message.chat.id, action="typing")
    try:
        state, latency = await asyncio.to_thread(run_agent_timed, query)
        answer = state.get("answer", "Нет ответа")
        intent = state.get("intent", "UNKNOWN")
        sgr = state.get("sgr_result", {})
        reply = answer
        if sgr:
            if sgr.get("found") is False:
                reply += "\n\nТочный ответ не найден"
            if sgr.get("citations"):
                cites = ", ".join(sgr["citations"][:5])
                reply += f"\n\nИсточники: {cites}"
            if sgr.get("defs"):
                reply += f"\n\nОпределения: {', '.join(sgr['defs'][:2])}"
        reply += f"\n\n_Intent: {intent} | {latency}s_"
        await message.answer(reply, parse_mode="Markdown")
        logger.info(f"{user_id}: {intent} {latency}s")
    except Exception as e:
        logger.error(f"{user_id}: {e}")
        await message.answer("Ошибка. Попробуй позже.")

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())