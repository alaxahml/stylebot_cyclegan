import logging
from torchvision.utils import save_image
from style_network.generator import Generator
import torchvision.transforms as transforms
import torch
from config_data.config import Config, load_config
from PIL import Image

#Создание логгера для дебага
logger = logging.getLogger(__name__)
config: Config = load_config()


def load_checkpoint(checkpoint_file: str, model: Generator, optimizer, lr: str, device: str) -> None:
    """
    Функция принимает файл с обученным состоянием сетки, объект сети и оптимизатора, а также learning rate для оптимизатора

    """
    logger.debug("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def initializing():
    inference_img = Image.open(config.content_img_path).convert("RGB")
    inference_tr = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    gen_AB = Generator()
    logger.debug("genAB created")
    gen_BA = Generator()
    logger.debug("genBA created")

    opt_gen = torch.optim.Adam(list(gen_AB.parameters()) + list(gen_BA.parameters()), lr=0.00001, betas=(0.5, 0.999))

    logger.debug("network created")
    load_checkpoint(
        config.chkpnt_genAB,
        gen_AB,
        opt_gen,
        0.00001,
        config.device
    )
    load_checkpoint(
        config.chkpnt_genBA,
        gen_BA,
        opt_gen,
        0.00001,
        config.device
    )

    logger.debug("weights loaded")

    inference_img = inference_tr(inference_img)

    return inference_img, gen_AB, gen_BA


def inference(gen_AB: Generator, gen_BA: Generator, img: torch.Tensor, device: str, direction="AB") -> None:
    """
    Функция непосредственной стилизации тестовых изображений- картинки прогоняются через тот или другой генератор,
    в зависимости от направления (из домена A в домен B, или наоборот).

    """
    with torch.no_grad():
        img = img.to(device)
        if direction == "AB":
          img = gen_AB(img)
        else:
          img = gen_BA(img)
        save_image(img * 0.5 + 0.5, config.generated_img_name)

def main_calling() -> None:
    """
    Функция предобрабатывает присланное в бот фото, создаёт объекты генераторов, подгружает из файлов обученные состояния
    генераторов, прогоняет через генераторы картинки и сохраняет их на диске.
    Дискриминаторы в боте не используются, т.к. они не нужны на инференсе
    """
    logger.debug("in main_calling")
    inference_img, gen_AB, gen_BA = initializing()
    inference(gen_AB, gen_BA, inference_img, config.device, "BA")



