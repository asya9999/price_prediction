#!/usr/bin/env python
# pylint: disable=W0613, C0116
# type: ignore[union-attr]
# This program is dedicated to the public domain under the CC0 license.

"""
Basic example for a bot that uses inline keyboards.
"""
import logging
from model import predict_price
from model2 import predict_price_DL
import model

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    Filters,
    ConversationHandler,
    CallbackContext,
    CallbackQueryHandler
)

ADD_INFO, PREDICT, BUTTON_HANDLER = range(3)

identity = ""

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

# Add normal text
def replay(callback):
    return {
        '1': "Type the *name* you your product",
        '2': "Specify condition of product from 1 to 5 \n*1 \- new product* and *5 \- poor quality product*",
        '3': "Set the *name of the categories* to which each product belongs \nYou can specify *up to 3* categories, separate them by */*",
        '4': "Type *brand name* of the product",
        '5': "Specify *delivery* details \n*0* \- the shipping fee is paid *by buyer*,  *1* \- paid *by seller*",
        '6': "Provide a detailed *description* of product",
    }[callback]

def id_element(callback):
    return {
        '1': "name",
        '2': "item_condition_id",
        '3': "category_name",
        '4': "brand_name",
        '5': "shipping",
        '6': "item_description",
        '7': "predict",
    }[callback]


def start(update: Update, context: CallbackContext) -> None:
    global cond
    keyboard = [
        [
            InlineKeyboardButton("name", callback_data='1'),
            InlineKeyboardButton("condition", callback_data='2'),
            InlineKeyboardButton("categories", callback_data='3'),
        ],
        [
            InlineKeyboardButton("brand", callback_data='4'),
            InlineKeyboardButton("shipping", callback_data='5'),
            InlineKeyboardButton("description", callback_data='6')
        ]
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)

    if cond:
        update.message.reply_text("Hi, I am a Bot developed by @aasya \U0001F430 and @alshch23 \U0001F42D for PMLDL course.\nI can predict prices of products \U0001F457 \U0001F45F \U0001F48D you are going to sell!")
    # Shows menu wuth buttons
    update.message.reply_text('Please specify information about product:', reply_markup=reply_markup)

    return BUTTON_HANDLER

def button(update: Update, context: CallbackContext) -> None:

    global identity


    query = update.callback_query
    query.answer()
    identity = id_element(query.data)

    print(identity)
    if identity=="predict":
        return ConversationHandler.END
    else:
        query.edit_message_text(text=f"{replay(query.data)}", parse_mode='MarkdownV2')
        return ADD_INFO


cond = True


def model_prediction(data):

    global to_model

    if data["name"]!=None and data["item_condition_id"]!=None and data["shipping"]!=None:
        #res = predict_price(data)
        res = predict_price(data)

        to_model = {"name": None,
            "item_condition_id": None,
            "category_name": None,
            "brand_name": None,
            "shipping": None,
            "item_description": None
}
        return res
    else:
        return "Not enough DATA!!!"


def predict(update: Update, context: CallbackContext) -> None:
    global cond
    global to_model

    update.message.reply_text("*Prediction result:*", parse_mode='MarkdownV2')
    
    pred_res = model_prediction(to_model)
    update.message.reply_text(f"{pred_res}")
    update.message.reply_text(form_product_info(to_model))
    
    cond = False
    keyboard = [
        [
            InlineKeyboardButton("name", callback_data='1'),
            InlineKeyboardButton("condition", callback_data='2'),
            InlineKeyboardButton("categories", callback_data='3'),
        ],
        [
            InlineKeyboardButton("brand", callback_data='4'),
            InlineKeyboardButton("shipping", callback_data='5'),
            InlineKeyboardButton("description", callback_data='6')
        ]
    ]

    #res = replay(keyboard)
    reply_markup = InlineKeyboardMarkup(keyboard)

    # Shows menu wuth buttons
    update.message.reply_text('Please specify information about product:', reply_markup=reply_markup)

    return BUTTON_HANDLER

def help_command(update: Update, context: CallbackContext) -> None:
    update.message.reply_text("To predict prices for you product just start the price predictor using /start")


to_model = {"name": None,
            "item_condition_id": None,
            "category_name": None,
            "brand_name": None,
            "shipping": None,
            "item_description": None
}

def check_input(id, message):
    if id=="shipping":
        try:
            if int(message) in [0, 1]:
                return int(message)
            else:
                return None 
        except:
            return None 

    elif id=="item_condition_id":

        try:
            if int(message) in range(1, 6):
                return int(message)
            else:
                return None 
        except:
            return None 
    else:
        return message


def form_product_info(product):
    output_string = "Your product information: \n"
    for key in product.keys():
        string = "" + str(key)+ ": " + str(product[key]) + "\n"
        output_string += string
    return output_string


def add_item(update: Update, context: CallbackContext) -> None:

    global identity

    text_to_write = check_input(identity, update.message.text)

    if text_to_write==None:
        update.message.reply_text("*Incorrect format*", parse_mode='MarkdownV2')
    else:
        to_model[identity] = update.message.text

    update.message.reply_text(form_product_info(to_model))
    
    keyboard = [
        [
            InlineKeyboardButton("name", callback_data='1'),
            InlineKeyboardButton("condition", callback_data='2'),
            InlineKeyboardButton("categories", callback_data='3'),
        ],
        [
            InlineKeyboardButton("brand", callback_data='4'),
            InlineKeyboardButton("shipping", callback_data='5'),
            InlineKeyboardButton("description", callback_data='6')
        ],
    ]

    #res = replay(keyboard)
    reply_markup = InlineKeyboardMarkup(keyboard)

    # Shows menu wuth buttons
    update.message.reply_text('Please specify information about product:', reply_markup=reply_markup)
    
    return BUTTON_HANDLER



def done(update: Update, context: CallbackContext) -> int:

    return ConversationHandler.END




def main():
    # Create the Updater and pass it your bot's token.
    updater = Updater("")


    # Add conversation handler with the states CHOOSING, TYPING_CHOICE and TYPING_REPLY
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            ADD_INFO: [
                MessageHandler(
                    Filters.text, add_item
                )
            ],
            BUTTON_HANDLER:
            [
                CallbackQueryHandler(button)
            ],
            PREDICT: [
                CallbackQueryHandler(
                    predict
                ),
            ]

            
        },
        fallbacks=[MessageHandler(Filters.regex('^Done$'), done)],
    )

    updater.dispatcher.add_handler(conv_handler)

    #updater.dispatcher.add_handler(CommandHandler('start', start))
    updater.dispatcher.add_handler(CommandHandler('predict_price', predict))
    #updater.dispatcher.add_handler(CallbackQueryHandler(button))
    updater.dispatcher.add_handler(CommandHandler('predict_price_info', help_command))

    # Start the Bot
    updater.start_polling()

    # Run the bot until the user presses Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT
    updater.idle()


if __name__ == '__main__':
    main()