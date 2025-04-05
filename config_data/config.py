from dataclasses import dataclass
from environs import Env
import torch

@dataclass()
class TgBot:
    token: str


@dataclass()
class Config:
    tg_bot: TgBot
    generated_img_name: str
    content_img_path: str
    device: str
    chkpnt_genAB: str
    chkpnt_genBA: str



def load_config(path=None) -> Config:
    """
    Функция загружает в класс Config необходимую информацию с окружения. Использовать её можно в боте, обращаясь к объекту класса Config, 
    возвращенному с данной функции
    """
    env = Env()
    env.read_env()
    config = Config(
        tg_bot=TgBot(token=env('BOT_TOKEN')),
        generated_img_name="generated.jpg",
        content_img_path="./content.jpg",
        device="cuda" if torch.cuda.is_available() else "cpu",
        chkpnt_genAB="genab.pth.tar",
        chkpnt_genBA="genba.pth.tar",

    )
    return config

