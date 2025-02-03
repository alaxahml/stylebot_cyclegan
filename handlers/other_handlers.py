import logging
from aiogram import Router
from aiogram.types import Message
from lexicon.lexicon import LEXICON_RU

#Создание роутера для работы с апдейтами и логгера для дебага
other_router = Router()
logger = logging.getLogger(__name__)


@other_router.message()
async def process_help_command(message: Message):
    """
    Обработка любых апдейтов, не попавших в user-хэндлеры

    """
    await message.answer(
        text=LEXICON_RU["other"]
    )
