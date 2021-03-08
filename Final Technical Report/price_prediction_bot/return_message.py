import re
import bs4
import requests


import pandas as pd 

to_model = {"name": "",
            "item_condition_id": "",
            "category_name": "",
            "brand_name": "",
            "shipping": "",
            "item_description": ""}

def text(input_text):

    counter = -1
    replay = ""

    if input_text=="/start":
        replay = "Hi, I am a Bot developed by @aasya and @alshch23 for PMLDL course.\nI can predict prices for products that you want to sell!"

    elif input_text=="/predict_price_info":
        replay = "To predict prices for you product you need to specify its name, condition, shipping. If you want - specify category, brand and description. To do that send comand /name, read instructions ans send values"

    elif input_text=="/name":
        replay = "In next message specify the name of product"
        counter = 0

    elif input_text=="/predict_price" or (counter >= 0): 
        if counter==0:
            replay = "In next messages specify, product charateristics \n\nWhat is the name of product?"
        elif counter==1:
            to_model["name"] = input_text
            replay = "What is item condition from 1 to 4 (1-excellent, 4-worse)?"
        elif counter==2:
            to_model["item_condition_id"] = int(input_text)
            replay = "Could you specify up to 3 categries, separating them by comma? (Mobile Phones, House)"
        elif counter==3:
            to_model["category_name"] = int(input_text)
            replay = "Could you specify up to 3 categries, separating them by comma? (Mobile Phones, House)"

        counter +=1
    return replay
