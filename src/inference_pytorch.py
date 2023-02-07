import os
import torch
from torch import nn
from torchvision import transforms
from PIL import Image

if __name__ == "__main__":

    crop_size = 224

    test_transforms = transforms.Compose(
        [
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485], [0.229]),
        ]
    )

    main_path = "sample_data/"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_ae = torch.jit.load("model/model_ae.pt")
    model_ae.to(device)
    model_ae.eval()
    criterion = nn.MSELoss()

    model_pred = torch.jit.load("model/model_pytorch_alexnet.pt")
    model_pred.to(device)
    model_pred.eval()

    ext = ("JPG", "jpg", "jpeg")
    for subdir, dirs, files in os.walk(main_path):
        # print(subdir)
        for i in files:
            if i.endswith(ext) and not i.startswith("."):
                img = Image.open(subdir + i).convert("RGB")

                img = test_transforms(img).to(device)
                img = torch.unsqueeze(img, 0)

                recon = model_ae(img)

                loss_ae = criterion(img, recon).item()

                if loss_ae < 0.0400:
                    pred = model_pred(img)
                    prob_softmax = torch.softmax(pred, dim=1)
                    _, standard_prediction = torch.max(prob_softmax, 1)
                    print(i)
                    print("BCS:", standard_prediction.detach().cpu().numpy()[0] + 2)
                else:
                    print("please use a different image")
