import logging
from torchvision.utils import save_image
from style_network.discriminator import Discriminator
from style_network.generator import Generator
import torchvision.transforms as transforms
import torch
from PIL import Image


logger = logging.getLogger(__name__)


def load_checkpoint(checkpoint_file, model, optimizer, lr, device):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def inference(gen_AB, gen_BA, img, device, direction="AB"):
  with torch.no_grad():
    img = img.to(device)
    if direction == "AB":
      img = gen_AB(img)
    else:
      img = gen_BA(img)
    save_image(img * 0.5 + 0.5, f"generated.jpg")

def main_calling():
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
    discr_A = Discriminator()
    logger.debug("discra created")
    discr_B = Discriminator()
    logger.debug("discrb created")

    opt_gen = torch.optim.Adam(list(gen_AB.parameters()) + list(gen_BA.parameters()), lr=0.00001, betas=(0.5, 0.999))
    opt_discr = torch.optim.Adam(list(discr_A.parameters()) + list(discr_B.parameters()), lr=0.00001,
                                 betas=(0.5, 0.999))

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
    load_checkpoint(
        "discra17.pth.tar",
        discr_A,
        opt_discr,
        0.00001,
        device
    )
    load_checkpoint(
        "discrb17.pth.tar",
        discr_B,
        opt_discr,
        0.00001,
        device
    )

    logger.debug("weights loaded")
    inference_img = inference_tr(inference_img)
    inference(gen_AB, gen_BA, inference_img, device,"BA")
    return True


