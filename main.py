#import gdown
import asyncio
import logging

from aiogram import Bot, Dispatcher
from config_data.config import Config, load_config
from handlers.user_handlers import user_router


# Настраиваем базовую конфигурацию логирования
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] #%(levelname)-8s %(filename)s:'
           '%(lineno)d - %(name)s - %(message)s'
)


# url_discra = "https://drive.google.com/uc?id=1oLcVEUAD2H8H91BqEaUhKCbYMh7Pbj_u"
# url_discrb = "https://drive.google.com/uc?id=1uZiTtihMgZq7ZUlO4w6tp4vPAdOo9MgS"
# url_genab = "https://drive.google.com/uc?id=1skcw9Q_zptM5KvRBoaCBYxqaWNL5F2-G"
# url_genba = "https://drive.google.com/uc?id=1oIWPp_Y9I0zWWpc19dhzrwrXmLobYpkf"
# gdown.download(url_discra)
# gdown.download(url_discrb)
# gdown.download(url_genab)
# gdown.download(url_genba)

# Инициализируем логгер модуля
logger = logging.getLogger(__name__)


# Функция конфигурирования и запуска бота
async def main() -> None:

    # Загружаем конфиг в переменную config
    config: Config = load_config()

    # Данные для передачи в хэндлеры
    nn_available = {"nn_available": True}


    # Инициализируем бот и диспетчер
    bot = Bot(token=config.tg_bot.token)
    dp = Dispatcher(nn_available=nn_available)

    # Регистрируем роутеры в диспетчере
    dp.include_router(user_router)


    # Запускаем polling
    await dp.start_polling(bot)


asyncio.run(main())