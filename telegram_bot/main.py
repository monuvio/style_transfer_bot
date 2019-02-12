from telegram.ext import Updater, CommandHandler, MessageHandler, RegexHandler
from telegram.ext import ConversationHandler, CallbackQueryHandler, Filters
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram_token import token
import logging

import torchvision.transforms as transforms
from torch.autograd import Variable
import torch

from array import array
from io import BytesIO
from PIL import Image
import numpy as np
import time
import os
import io

from model_1 import transfer_style
from model_2 import transform



logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO)

logger = logging.getLogger(__name__)

MENU, SET_STAT, ABOUT, NEURO, NEURO_1, NEURO_PREP = range(6)


def start(bot, update):
    """
    Start function. Calls when the /start command is called.
    """
    return menu(bot, update)


def menu(bot, update):
    """
    Main menu function.
    This will display the options from the main menu.
    """
    keyboard = [["Перенос стиля", "О боте"]]
    reply_markup = ReplyKeyboardMarkup(keyboard,
                                       one_time_keyboard=True,
                                       resize_keyboard=True)
    user = update.message.from_user
    logger.info("Menu command shows for {}.".format(user.first_name))
    update.message.reply_text("Привет! Я бот для переноса стилей! \n"\
                                "Выбирай одну из функций ниже", reply_markup=reply_markup)
    return SET_STAT


def set_state(bot, update):
    """
    Set option selected from menu.
    """
    user = update.message.from_user
    if update.message.text == "Перенос стиля":
        return neural_set(bot, update)
    elif update.message.text == "О боте":
        return about_bot(bot, update)
    elif update.message.text == "Свои стили":
        return model_01(bot, update)
    elif update.message.text == "Готовые стили":
        return model_02_prep(bot, update)
    else:
        return MENU
        
def model_01(bot, update):
    """
    Reply instruction message for first styling function and return this function with photo handler.
    """
    user = update.message.from_user
    logger.info("First style requested by {}.".format(user.first_name))
    update.message.reply_text(
        "Загрузи 2 картинки: сначала картинку, с которой нейросеть возьмет объект," \
        "затем картинку, с которой нейросеть возьмет стиль и применит к первой картинке.")
    return NEURO
    
def model_02_prep(bot, update):
    """
    Show required styles and return style choice function with text handler.
    """
    required = ['Металл', 'Кубизм']
    update.message.reply_text(
        "Напиши стиль, который хочешь применить к изображению" \
        "\n Доступные стили: {}".format(', '.join(required)))
    return NEURO_PREP   
    
style = ''

def model_02(bot, update):
    """
    Redirecting on second styling function
    """
    user = update.message.from_user
    logger.info("Second style requested by {}.".format(user.first_name))
    global style
    if update.message.text == "Металл":
        style = 'steel'
    elif update.message.text == "Кубизм":
        style = 'cubism'
    else:
        update.message.reply_text("Нет такого стиля")
    update.message.reply_text(
        "Теперь отправь изображение, на которое наложится стиль.")
    return NEURO_1
    
first_image_file = {}
        
def choose_style(bot, update):
    """
    Main function of second style transfer. Takes content image and send result of transfer.
    """
    chat_id = update.message.chat_id
    print("Got image from {}".format(chat_id))

    image_info = update.message.photo[-1]
    image_file = bot.get_file(image_info) 

    content_image_stream = BytesIO()
    image_file.download(out=content_image_stream)
    
    output = transform('/content/{}.pth'.format(style), content_image_stream)    
        
    output_stream = BytesIO()
    output.save(output_stream, format='PNG')
    output_stream.seek(0)
    bot.send_photo(chat_id, photo=output_stream)
    print("Sent Photo to user")
    update.message.reply_text("Ты можешь вернуться обратно в меню с помощью команды /menu")

    return MENU


def send_prediction_on_photo(bot, update):
    """
    Main function of first style transfer. Takes content and style images and send result of transfer.
    """
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
        update.message.reply_text("Изображения получены. Подожди примерно минуту.")

        output = transfer_style(content_image_stream, style_image_stream)

        output_stream = BytesIO()
        output.save(output_stream, format='PNG')
        output_stream.seek(0)
        bot.send_photo(chat_id, photo=output_stream)
        print("Sent Photo to user")
        update.message.reply_text("Ты можешь вернуться обратно в меню с помощью команды /menu")

        return MENU
    else:
        first_image_file[chat_id] = image_file
        return NEURO


def neural_set(bot, update):
    """
    Redirecting on styling function
    """
    keyboard = [["Свои стили", "Готовые стили"]]
    reply_markup = ReplyKeyboardMarkup(keyboard,
                                       one_time_keyboard=True,
                                       resize_keyboard=True)
    user = update.message.from_user
    logger.info("Neural style requested by {}.".format(user.first_name))
    update.message.reply_text('Каким образом ты хочешь наложить стиль на картинку?' \
    '\n У каждого из методов есть отличие.' \
    'Если не знаешь, чем они отличаются, то вызови /menu и зайди в раздел "О боте".', reply_markup=reply_markup)
    return SET_STAT


def about_bot(bot, update):
    """
    About function. Displays info about Style Transfer Bot.
    """
    user = update.message.from_user
    logger.info("About info requested by {}.".format(user.first_name))
    bot.send_message(chat_id=update.message.chat_id, text=
    """
    Методы переноса стиля: 
    1. 'Свои стили'. 
    Этот вариант дает возможность использовать свои изображения стиля. 
    (Работает гораздо медленнее, нежели второй вариант, но более качественно.)
    2. 'Готовые стили'.
    Этот вариант использует готовые стили. 
    За несколько секунд перенесет стили, которые создатель посчитал интересными.
    Список стилей и примеры работы можно найти на сайте: ###
    
    Этот бот создан в рамках проектной работы в Deep learning school.
    """)
    bot.send_message(chat_id=update.message.chat_id, text="Ты можешь вернуться обратно в меню с помощью команды /menu.")
    return MENU


def help(bot, update):
    """
    Help function.
    This displays a set of commands available for the bot.
    """
    user = update.message.from_user
    logger.info("User {} asked for help.".format(user.first_name))
    update.message.reply_text(
        "Используй команду /cancel , чтобы выйти из чата. \nИспользуй /start , чтобы перезагрузить бота.",
        reply_markup=ReplyKeyboardRemove())


def cancel(bot, update):
    """
    User cancelation function.
    Cancel conversation by user.
    """
    user = update.message.from_user
    logger.info("User {} canceled the conversation.".format(user.first_name))
    update.message.reply_text("Пока! Надеемся пообщаться с тобой ещё!",
                              reply_markup=ReplyKeyboardRemove())

    return ConversationHandler.END


def error(bot, update, error):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, error)


def main():
    """
    Main function.
    This function handles the conversation flow by setting
    states on each step of the flow. Each state has its own
    handler for the interaction with the user.
    """

    # Create the EventHandler and pass it your bot's token.
    updater = Updater(token, request_kwargs={'proxy_url': 'socks4://5.196.59.57:30248'})

    # Get the dispatcher to register handlers:
    dp = updater.dispatcher

    # Add conversation handler with predefined states:
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],

        states={
            MENU: [CommandHandler('menu', menu)],
            
            NEURO_PREP: [MessageHandler(Filters.text, model_02)],

            NEURO: [MessageHandler(Filters.photo, send_prediction_on_photo)],
            
            NEURO_1: [MessageHandler(Filters.photo, choose_style)],

            SET_STAT: [RegexHandler(
                '^({}|{})$'.format(
                "Перенос стиля", "О боте"),
                set_state), 
               RegexHandler(
               '^({}|{})$'.format(
               "Свои стили", "Готовые стили"),
               set_state)]
        },

        fallbacks=[CommandHandler('cancel', cancel),
                   CommandHandler('help', help)],
                   
        conversation_timeout = 900.0
    )

    dp.add_handler(conv_handler)

    # Log all errors:
    dp.add_error_handler(error)

    # Start Style transfer bot:
    updater.start_polling()

    # Run the bot until the user presses Ctrl-C or the process
    # receives SIGINT, SIGTERM or SIGABRT:
    updater.idle()


if __name__ == '__main__':
    print('ready')
    main()