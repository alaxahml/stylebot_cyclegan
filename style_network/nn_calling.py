import logging
from torchvision.utils import save_image
from style_network.generator import Generator
import torchvision.transforms as transforms
import torch
from PIL import Image

#Создание логгера для дебага
logger = logging.getLogger(__name__)


def load_checkpoint(checkpoint_file, model, optimizer, lr, device):
    """
    Функция принимает файл с обученным состоянием сетки, объект сети и оптимизатора, а также learning rate для оптимизатора

    """
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def inference(gen_AB, gen_BA, img, device, direction="AB"):
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
        save_image(img * 0.5 + 0.5, f"generated.jpg")

def main_calling():
    """
    Функция предобрабатывает присланное в бот фото, создаёт объекты генераторов, подгружает из файлов обученные состояния
    генераторов, прогоняет через генераторы картинки и сохраняет их на диске.
    Дискриминаторы в боте не используются, т.к. они не нужны на инференсе
    """
    logger.debug("in main_calling")
    inference_img = Image.open("./content.jpg").convert("RGB")
    inference_tr = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen_AB = Generator()
    logger.debug("genab created")
    gen_BA = Generator()
    logger.debug("genba created")

    opt_gen = torch.optim.Adam(list(gen_AB.parameters()) + list(gen_BA.parameters()), lr=0.00001, betas=(0.5, 0.999))


    logger.debug("network created")
    load_checkpoint(
        "genab17.pth.tar",
        gen_AB,
        opt_gen,
        0.00001,
        device
    )
    load_checkpoint(
        "genba17.pth.tar",
        gen_BA,
        opt_gen,
        0.00001,
        device
    )

    logger.debug("weights loaded")
    inference_img = inference_tr(inference_img)
    inference(gen_AB, gen_BA, inference_img, device,"BA")



