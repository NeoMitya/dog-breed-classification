import os
import json
import asyncio
from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart
from ai import predict_breed
from decouple import config

BOT_TOKEN = config("TELEGRAM_TOKEN")

dp = Dispatcher()
bot = Bot(token=BOT_TOKEN)

with open("model/breeds_rus.json", "r", encoding="utf-8") as f:
    BREEDS = json.load(f)


@dp.message(CommandStart())
async def start_cmd(message: types.Message):
    await message.answer("Привет! 🐶 Пришли мне фото собаки, и я попробую определить её породу!")


@dp.message(lambda msg: msg.photo)
async def handle_photo(message: types.Message):
    photo = message.photo[-1]
    path = f"temp_{message.from_user.id}.jpg"

    file_info = await bot.get_file(photo.file_id)
    await bot.download_file(file_info.file_path, destination=path)

    top3 = predict_breed(path)
    os.remove(path)

    text = "Породы, которые я вижу:\n"
    for idx, prob in top3:
        text += f"• {BREEDS[str(idx)]} — {prob * 100:.1f}%\n"

    await message.answer(text)


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
