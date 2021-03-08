import pickle 

import pandas as pd 

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

import numpy as np
import tensorflow as tf

# Model used name
model_name = "./linear_model.pkl"
with open(model_name, 'rb') as file:
    prediction_model = pickle.load(file)

# loading
with open('tokenizer.pickle', 'rb') as handle:
    result_tokenizer = pickle.load(handle)

with open('lb_brand_name.pkl', 'rb') as f:
    lb_brand_name = pickle.load(f)


#load DL model
# loading the model architecture and asigning the weights
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model_loaded = tf.keras.models.model_from_json(loaded_model_json)
# load weights into new model
model_loaded.load_weights("model_weights.h5")

# parameters

missing_value = "missing"
missing_cat_value = "missing"
categ_length = 3

#input to model
MAX_NAME_SEQ = 20
MAX_ITEM_DESC_SEQ = 60
MAX_CAT1_NAME_SEQ = 20
MAX_CAT2_NAME_SEQ = 20
MAX_CAT3_NAME_SEQ = 20
MAX_BRAND_NAME_SEQ = 20


def handle_missing(input_dataset):
    def process_descr(text, t):
      if isinstance(text, str) and text!=t:
        return text
      else:
        return missing_value

    dataset = input_dataset.copy()
    dataset['category_name'].fillna(value = missing_value, inplace=True)
    dataset['brand_name'].fillna(value = missing_value, inplace=True)
    no_descr = "No description yet"
    dataset['item_description'] = dataset['item_description'].apply(lambda x: process_descr(x, no_descr))
    dataset['item_description'].fillna(value = missing_value, inplace=True)
    return dataset

def handle_cateogry(input_dataset):
  def split_category_name(text):
    try:
      new_text = text.split("/")
      while len(new_text) < categ_length:
        new_text.append(missing_cat_value)
      return new_text[:categ_length]
    except:
      return [missing_cat_value, missing_cat_value, missing_cat_value]

  dataset = input_dataset.copy()
  res = dataset['category_name'].apply(lambda x: split_category_name(x))
  dataset['cat_1'] = dataset['category_name'].apply(lambda x: split_category_name(x)[0])
  dataset['cat_2'] = dataset['category_name'].apply(lambda x: split_category_name(x)[1])
  dataset['cat_3'] = dataset['category_name'].apply(lambda x: split_category_name(x)[2])
  return dataset


def tokenize(input_dataset, tokenizer):
  # add padding and cropping( in case its longer)
  dataset = input_dataset.copy()
  dataset["seq_cat_1_name"] = tokenizer.texts_to_sequences(dataset.cat_1.str.lower())
  dataset["seq_cat_2_name"] = tokenizer.texts_to_sequences(dataset.cat_2.str.lower())
  dataset["seq_cat_3_name"] = tokenizer.texts_to_sequences(dataset.cat_3.str.lower())
  dataset["seq_brand_name"] = tokenizer.texts_to_sequences(dataset.brand_name.str.lower())
  dataset["seq_item_description"] = tokenizer.texts_to_sequences(dataset.item_description.str.lower())
  dataset["seq_name"] = tokenizer.texts_to_sequences(dataset.name.str.lower())
  return dataset


def make_tokenizer(input_dataset):
  raw_text = np.hstack([input_dataset.cat_1.str.lower(), 
                        input_dataset.cat_2.str.lower(), 
                        input_dataset.cat_3.str.lower(), 
                        input_dataset.brand_name.str.lower(), 
                        input_dataset.item_description.str.lower(), 
                        input_dataset.name.str.lower()])
  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(raw_text)
  return tokenizer


def form_input_to_model(dataset, lb_brand_name):
    X = {
        'name': pad_sequences(dataset.seq_name, maxlen=MAX_NAME_SEQ)
        ,'item_desc': pad_sequences(dataset.seq_item_description
                                    , maxlen=MAX_ITEM_DESC_SEQ)
        ,'cat_1': pad_sequences(dataset.seq_cat_1_name
                                        , maxlen=MAX_CAT1_NAME_SEQ)
        ,'cat_2': pad_sequences(dataset.seq_cat_2_name
                                        , maxlen=MAX_CAT2_NAME_SEQ)
        ,'cat_3': pad_sequences(dataset.seq_cat_3_name
                                        , maxlen=MAX_CAT3_NAME_SEQ)
        
        ,'brand_name': pad_sequences(dataset.seq_brand_name
                                        , maxlen=MAX_BRAND_NAME_SEQ)
        ,'item_condition': np.array([int(i) for i in dataset.item_condition_id])
        ,'shipping': np.array([int(i) for i in dataset.shipping])
    }
    return X


def prepare_test(dataset):
    global result_tokenizer
    new_train = handle_missing(dataset)
    new_train = handle_cateogry(new_train)
    new_train = tokenize(new_train, result_tokenizer)
    X = form_input_to_model(new_train, lb_brand_name)

    return X


def predict_price(data):

    global prediction_model

    for key in data.keys():
        data[key] = [data[key]]
    df = pd.DataFrame(data) 

    X_test_new = prepare_test(df)

    print(X_test_new)

    X_matrix_test = (X_test_new['name'],
            X_test_new['item_desc'],
            X_test_new['cat_1'],
            X_test_new['cat_2'],
            X_test_new['cat_3'],
            X_test_new['brand_name'],            
            np.array([[i] for i in X_test_new['item_condition']]),
            np.array([[i] for i in X_test_new['shipping']])
            )

    X_matrix_test = np.hstack(X_matrix_test)
    print("!!!!!!!!!!!!!!!")
    print(X_matrix_test)

    result = np.expm1(prediction_model.predict(X_matrix_test))
    return result[0]
    
def predict_price_DL(data):

    global prediction_model

    for key in data.keys():
        data[key] = [data[key]]
    df = pd.DataFrame(data) 

    X_test_new = prepare_test(df)

    print(X_test_new)


    result = np.expm1(model_loaded.predict(X_test_new))
    return result[0]
    


