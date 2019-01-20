from telegram.ext import Updater, CommandHandler, MessageHandler, RegexHandler
from telegram.ext import ConversationHandler, CallbackQueryHandler, Filters
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove
from lang_dict import *
import logging
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

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO)

logger = logging.getLogger(__name__)

LANG = "EN"
SET_LANG, MENU, SET_STAT, ABOUT, NEURO, NEURO_STAT = range(6)
STATE = SET_LANG

def start(bot, update):
    """
    Start function. Displayed whenever the /start command is called.
    This function sets the language of the bot.
    """
    # Create buttons to select language:
    keyboard = [['RU', 'EN']]

    # Create initial message:
    message = "Hey, i'm 'Style transfer bot! / Привет! Я бот для переноса стилей! \n\n\
Please select a language to start. / Пожалуйста, выбери язык, чтобы начать."
				

    reply_markup = ReplyKeyboardMarkup(keyboard,
                                       one_time_keyboard=True,
                                       resize_keyboard=True)
    update.message.reply_text(message, reply_markup=reply_markup)

    return SET_LANG
	
def set_lang(bot, update):
    """
    First handler with received data to set language globally.
    """
    # Set language:
    global LANG
    LANG = update.message.text
    user = update.message.from_user

    logger.info("Language set by {} to {}.".format(user.first_name, LANG))
    update.message.reply_text(lang_selected[LANG],
                              reply_markup=ReplyKeyboardRemove())

    return MENU
	
def menu(bot, update):
    """
    Main menu function.
    This will display the options from the main menu.
    """
    # Create buttons to select language:
    keyboard = [[choose_net_variation[LANG], view_about[LANG]]]
    reply_markup = ReplyKeyboardMarkup(keyboard,
                                       one_time_keyboard=True,
                                       resize_keyboard=True)
    user = update.message.from_user
    logger.info("Menu command requested by {}.".format(user.first_name))
    update.message.reply_text(main_menu[LANG], reply_markup=reply_markup)
    return SET_STAT
	
def set_state(bot, update):
    """
    Set option selected from menu.
    """
    # Set state:
    global STATE
    user = update.message.from_user
    if update.message.text == choose_net_variation[LANG]:
        STATE = NEURO_STAT
        return neural_set(bot, update)
    elif update.message.text == view_about[LANG]:
        STATE = ABOUT
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
    update.message.reply_text(neural_net[LANG])
    return NEURO
    	
def about_bot(bot, update):
    """
    About function. Displays info about DisAtBot.
    """
    user = update.message.from_user
    logger.info("About info requested by {}.".format(user.first_name))
    bot.send_message(chat_id=update.message.chat_id, text=about_info[LANG])
    bot.send_message(chat_id=update.message.chat_id, text=back2menu[LANG])
    return MENU
	
def help(bot, update):
    """
    Help function.
    This displays a set of commands available for the bot.
    """
    user = update.message.from_user
    logger.info("User {} asked for help.".format(user.first_name))
    update.message.reply_text(help_info[LANG],
                              reply_markup=ReplyKeyboardRemove())
	
def cancel(bot, update):
    """
    User cancelation function.
    Cancel conversation by user.
    """
    user = update.message.from_user
    logger.info("User {} canceled the conversation.".format(user.first_name))
    update.message.reply_text(goodbye[LANG],
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
    global LANG
    # Create the EventHandler and pass it your bot's token.
    updater = Updater(token, request_kwargs={'proxy_url': 'socks4://168.195.171.42:44880'})

    # Get the dispatcher to register handlers:
    dp = updater.dispatcher

    # Add conversation handler with predefined states:
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],

        states={
            SET_LANG: [RegexHandler('^(RU|EN)$', set_lang)],

            MENU: [CommandHandler('menu', menu)],
			
            NEURO: [MessageHandler(Filters.photo, send_prediction_on_photo)],

            SET_STAT: [RegexHandler(
                        '^({}|{})$'.format(
                            choose_net_variation['RU'], view_about['RU'],),
                        set_state),
                       RegexHandler(
                        '^({}|{})$'.format(
                            choose_net_variation['EN'], view_about['EN'],),
                        set_state)],
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