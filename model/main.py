from telegram.ext import Updater, CommandHandler, MessageHandler, RegexHandler
from telegram.ext import ConversationHandler, CallbackQueryHandler, Filters
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove
import logging
from model import StyleTransferModel
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

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO)

logger = logging.getLogger(__name__)

MENU, SET_STAT, ABOUT, NEURO= range(4)

def start(bot, update):
    """
    Start function. Displayed whenever the /start command is called.
    This function sets the language of the bot.
    """
    update.message.reply_text("Привет! Я бот для переноса стилей!")
    return menu(bot, update)
	
def menu(bot, update):
    """
    Main menu function.
    This will display the options from the main menu.
    """
    keyboard = [["Нейросетевой перенос стиля", "О боте"]]
    reply_markup = ReplyKeyboardMarkup(keyboard,
                                       one_time_keyboard=True,
                                       resize_keyboard=True)
    user = update.message.from_user
    logger.info("Menu command requested by {}.".format(user.first_name))
    update.message.reply_text("Чтож, выбери одну из функций ниже", reply_markup=reply_markup)
    return SET_STAT
	
def set_state(bot, update):
    """
    Set option selected from menu.
    """
    # Set state:
    global STATE
    user = update.message.from_user
    if update.message.text == "Нейросетевой перенос стиля":
        return neural_set(bot, update)
    elif update.message.text == "О боте":
        return about_bot(bot, update)
    else:
        STATE = MENU
        return MENU
		
model = StyleTransferModel()
first_image_file = {}

def send_prediction_on_photo(bot, update):
    """
    Neural networks should work :)
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
		
        csF = torch.Tensor()
        csF = Variable()
        print('Transferring')
        start_time = time.time()

        output = model.transfer_style(content_image_stream, style_image_stream, csF)
		
        end_time = time.time()
        print('Elapsed time is: %f' % (end_time - start_time))

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
    return
		
def neural_set(bot, update):
    """
    Redirecting on styling function
    """
    user = update.message.from_user
    logger.info("Neural style requested by {}.".format(user.first_name))
    update.message.reply_text("Загрузи 2 картинки: сначала картинку, с которой нейросеть возьмет объект, затем картинку, с которой нейросеть возьмет стиль и применит к первой картинке.")
    return 
    	
def about_bot(bot, update):
    """
    About function. Displays info about DisAtBot.
    """
    user = update.message.from_user
    logger.info("About info requested by {}.".format(user.first_name))
    bot.send_message(chat_id=update.message.chat_id, text="Чатбот для переноса стилей##TODO \n\n\
")
    bot.send_message(chat_id=update.message.chat_id, text="Ты можешь вернуться обратно в меню с помощью команды /menu.")
    return MENU
	
def help(bot, update):
    """
    Help function.
    This displays a set of commands available for the bot.
    """
    user = update.message.from_user
    logger.info("User {} asked for help.".format(user.first_name))
    update.message.reply_text("Используй команду /cancel , чтобы выйти из чата. \nИспользуй /start , чтобы перезагрузить бота.",
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
    updater = Updater(token, request_kwargs={'proxy_url': 'socks4://148.251.113.238:50879'})

    # Get the dispatcher to register handlers:
    dp = updater.dispatcher

    # Add conversation handler with predefined states:
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],

        states={
            MENU: [CommandHandler('menu', menu)],
			
            NEURO: [MessageHandler(Filters.photo, send_prediction_on_photo)],

            SET_STAT: [RegexHandler(
                        '^({}|{})$'.format(
                            "Нейросетевой перенос стиля", "О боте",),
                        set_state)]
        },

        fallbacks=[CommandHandler('cancel', cancel),
                   CommandHandler('help', help)]
    )

    dp.add_handler(conv_handler)

    # Log all errors:
    dp.add_error_handler(error)

    # Start DisAtBot:
    updater.start_polling()

    # Run the bot until the user presses Ctrl-C or the process
    # receives SIGINT, SIGTERM or SIGABRT:
    updater.idle()


if __name__ == '__main__':
    print('ready')
    main()

