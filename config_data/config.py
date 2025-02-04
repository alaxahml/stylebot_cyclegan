from dataclasses import dataclass
from environs import Env

@dataclass()
class TgBot:
    token: str


@dataclass()
class Config:
    tg_bot: TgBot


def load_config(path=None) -> Config:
    """
    Функция загружает в класс Config необходимую информацию с окружения. Использовать её можно в боте, обращаясь к объекту класса Config, 
    возвращенному с данной функции
    """
    env = Env()
    env.read_env()
    return Config(tg_bot=TgBot(token=env('BOT_TOKEN')))

