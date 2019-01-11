from modelsus import StyleTransferModel
from telegram_token import token
import numpy as np
from PIL import Image
from io import BytesIO
from array import array
import os
import io
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import torch

model = StyleTransferModel()
first_image_file = {}

def send_prediction_on_photo(bot, update):
    chat_id = update.message.chat_id
    print("Got image from {}".format(chat_id))

    image_info = update.message.photo[-1]
    image_file = bot.get_file(image_info)

    if chat_id in first_image_file:

        content_image_stream = BytesIO()
        first_image_file[chat_id].download(out=content_image_stream)
        del first_image_file[chat_id]

        style_image_stream = BytesIO()
        image_file.download(out=style_image_stream)

        output = model.transfer_style(content_image_stream, style_image_stream)

        output_stream = BytesIO()
        unloader = transforms.ToPILImage()
        output = torch.reshape(output, [3, 128, 128])
        output = unloader(output)
        output.save(output_stream, format='PNG')
        output_stream.seek(0)
        bot.send_photo(chat_id, photo=output_stream)
        print("Sent Photo to user")
    else:
        first_image_file[chat_id] = image_file


