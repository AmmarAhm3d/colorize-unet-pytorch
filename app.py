import gradio as gr
from PIL import Image
import torch, numpy as np
from skimage.color import rgb2lab, lab2rgb
from skimage.transform import resize
from model import UNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 256

def load_model():
    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load("final_colorization_model.pth", map_location=DEVICE))
    model.eval()
    return model

model = load_model()

def colorize_image(image):
    img_resized = resize(np.array(image), (IMAGE_SIZE, IMAGE_SIZE))
    lab_img = rgb2lab(img_resized)
    l_channel = lab_img[:, :, 0]
    l_tensor = torch.from_numpy((l_channel / 50.0) - 1.0).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        predicted_ab = model(l_tensor)
    predicted_ab = predicted_ab.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 128.0

    lab_colorized = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3))
    lab_colorized[:, :, 0] = l_channel
    lab_colorized[:, :, 1:] = predicted_ab

    rgb_colorized = lab2rgb(lab_colorized)
    return (rgb_colorized * 255).astype(np.uint8)

demo = gr.Interface(
    fn=colorize_image,
    inputs=gr.Image(type="pil", label="Upload grayscale image"),
    outputs=gr.Image(label="Colorized image"),
    title="🎨 AI Image Colorizer",
)

demo.launch(share=True)
