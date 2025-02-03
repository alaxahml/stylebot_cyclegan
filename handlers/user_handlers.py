import asyncio
import logging
from aiogram import Bot
from aiogram import F, Router
from aiogram.filters import Command, CommandStart, StateFilter
from aiogram.fsm.state import default_state
from aiogram.fsm.context import FSMContext
from aiogram.types import Message, PhotoSize, FSInputFile
from fsm.fsm import FSMStyleGen
from style_network.nn_calling import main_calling
from concurrent.futures import ThreadPoolExecutor
from lexicon.lexicon import LEXICON_RU

#Создание пула потоков для асинхронности сервера
pool = ThreadPoolExecutor(max_workers=4)
#Создание роутера для работы с апдейтами и логгера для дебага
user_router = Router()
logger = logging.getLogger(__name__)


@user_router.message(Command(commands="help"))
async def process_help_command(message: Message):
    """
    Хэндлер принимает апдейт-команду /help, отправляет в бот соответствующее сообщение

    """
    await message.answer(
        text=LEXICON_RU["help"]
    )


@user_router.message(F.photo[-1].as_('largest_photo'),
                     StateFilter(FSMStyleGen.send_photo, FSMStyleGen.waiting_for_network))
async def process_photo_command(message: Message, largest_photo: PhotoSize,
                                state: FSMContext, bot: Bot, nn_available: dict):
    """
    Хэндлер принимает фото с чата, и, в случае верного FSM состояния бота, передаёт фото на обработку в нейронную сеть.
    Если нейросеть будет занята обработкой какого-либо другого фото, то пользователь получит соответствующее уведомление в чат.
    Все фото, попавшие в этот хэндлер, будут обработаны нейросетью . Доступность нейросети определяется флагом nn_available
    """

    file = await bot.get_file(largest_photo.file_id)
    photo_name = "content.jpg"

    logger.debug("in process_photo")

    if not nn_available["nn_available"]:
        await message.answer(
            text=LEXICON_RU["nn_unavailable"]
        )

    #Ожидание в случае недоступной нейросети
    while not nn_available["nn_available"]:
        await asyncio.sleep(1)

    #В случае прохода апдейта до этой строки, флаг доступности устанавливается в false - следующие фото будут ожидать доступность в цикле
    nn_available["nn_available"] = False

    await state.set_state(FSMStyleGen.waiting_for_network)
    #Скачивание фото, присланного из чата, будет взято нейросетью для обработки
    await bot.download_file(file.file_path, photo_name)

    logger.debug("nn begins")

    #Создание нового потока, вызов нейросети и ожидание окончания работы нейросети
    work_nn = pool.submit(main_calling)
    while not work_nn.done():
        await asyncio.sleep(1)

    image_from_nn = FSInputFile("generated.jpg")

    #Посылка результата
    await message.answer_photo(
        image_from_nn,
        caption=LEXICON_RU["result"]
    )

    #Установление состояния FSM и флага nn_available - теперь можно обрабатывать следующие фото
    await state.set_state(FSMStyleGen.send_photo)
    nn_available["nn_available"] = True


@user_router.message(StateFilter(FSMStyleGen.waiting_for_network))
async def process_any_during_network(message: Message):
    """
    Обработка произвольного апдейта во время работы нейросети.

    """
    await message.answer(
        text=LEXICON_RU["waiting_for_network"]
    )



@user_router.message(CommandStart(), StateFilter(default_state))
async def process_start_command(message: Message, state: FSMContext):
    """
    Обработка апдейта-команды /start в случае незапущенного бота

    """
    await message.answer(
        text=LEXICON_RU["start"]
    )

    await state.set_state(FSMStyleGen.send_photo)



@user_router.message(CommandStart())
async def process_wrong_start_command(message: Message):
    """
    Обработка апдейта-команды /start в случае запущенного бота

    """
    await message.answer(
        text=LEXICON_RU["wrong_start"]
    )


@user_router.message(Command(commands="cancel"), ~StateFilter(default_state))
async def process_cancel_command(message: Message, state: FSMContext):
    """
    Обработка апдейта-команды /cancel в случае несброшенного состояния бота. Сбрасывает состояние бота

    """
    await message.answer(
        text=LEXICON_RU["cancel"]
    )

    await state.clear()


@user_router.message(Command(commands="cancel"))
async def process_wrong_cancel_command(message: Message):
    """
    Обработка апдейта-команды /cancel в случае уже сброшенного состояния бота.

    """
    await message.answer(
        text=LEXICON_RU["wrong_cancel"]
    )


















