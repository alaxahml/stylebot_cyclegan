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

pool = ThreadPoolExecutor(max_workers=2)
user_router = Router()
logger = logging.getLogger(__name__)


@user_router.message(Command(commands="help"))
async def process_help_command(message: Message):
    await message.answer(
        text=LEXICON_RU["help"]
    )


@user_router.message(F.photo[-1].as_('largest_photo'),
                     StateFilter(FSMStyleGen.send_photo, FSMStyleGen.waiting_for_network))
async def process_photo_command(message: Message, largest_photo: PhotoSize,
                                state: FSMContext, bot: Bot, nn_available: dict):

    file = await bot.get_file(largest_photo.file_id)
    photo_name = "content.jpg"

    if not nn_available["nn_available"]:
        await message.answer(
            text=LEXICON_RU["nn_unavailable"]
        )

    while not nn_available["nn_available"]:
        pass

    nn_available["nn_available"] = False
    await state.set_state(FSMStyleGen.waiting_for_network)
    await bot.download_file(file.file_path, photo_name)

    logger.debug("nn begins")
    work_nn = pool.submit(main_calling)


    while not work_nn.done():
        pass


    image_from_nn = FSInputFile("generated.jpg")

    await message.answer_photo(
        image_from_nn,
        caption="Результат"
    )

    await state.clear()
    nn_available["nn_available"] = True



@user_router.message(StateFilter(FSMStyleGen.waiting_for_network))
async def process_any_during_network(message: Message):
    await message.answer(
        text=LEXICON_RU["waiting_for_network"]
    )



@user_router.message(CommandStart(), StateFilter(default_state))
async def process_start_command(message: Message, state: FSMContext):
    await message.answer(
        text=LEXICON_RU["start"]
    )

    await state.set_state(FSMStyleGen.send_photo)



@user_router.message(CommandStart())
async def process_wrong_start_command(message: Message):
    await message.answer(
        text=LEXICON_RU["wrong_start"]
    )


@user_router.message(Command(commands="cancel"), ~StateFilter(default_state))
async def process_cancel_command(message: Message, state: FSMContext):
    await message.answer(
        text=LEXICON_RU["cancel"]
    )

    await state.clear()


@user_router.message(Command(commands="cancel"))
async def process_wrong_cancel_command(message: Message):
    await message.answer(
        text=LEXICON_RU["wrong_cancel"]
    )


















