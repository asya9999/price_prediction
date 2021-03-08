#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Mount folder with Dataset
#from google.colab import drive
#drive.mount('/content/drive')


# In[ ]:


#cd '/content/drive/MyDrive/Price_prediction_project'


# In[ ]:


import pandas as pd 

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

import numpy as np


# In[ ]:


test = pd.read_csv("./test.tsv",sep='\t')
train = pd.read_csv("./train.tsv",sep='\t')


# In[ ]:


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


# In[ ]:


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


# In[ ]:


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


# In[ ]:


def make_y(data):
  return np.log1p(data)


# In[ ]:



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
        ,'item_condition': np.array(dataset.item_condition_id)
        ,'shipping': np.array(dataset.shipping)
    }
    return X


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train, train['price'], test_size=0.2)


# In[ ]:


# Final version for prepare dataset
def prepare_train(dataset):
    new_train = handle_missing(dataset)
    new_train = handle_cateogry(new_train)

    result_tokenizer = make_tokenizer(new_train)
    new_train = tokenize(new_train, result_tokenizer)

    #lb_brand_name = LabelBinarizer(sparse_output=True)
    lb_brand_name = LabelBinarizer()
    lb_brand_name = lb_brand_name.fit(new_train.brand_name)

    X = form_input_to_model(new_train, lb_brand_name)

    return X, result_tokenizer, lb_brand_name


# In[ ]:


X_train_new, result_tokenizer, lb_brand_name = prepare_train(X_train)


# In[ ]:


y_train_new = make_y(y_train)
y_train_new = np.array(y_train_new)


# In[ ]:


print(X_train_new["brand_name"])


# In[ ]:





# In[ ]:


def prepare_test(dataset):
    new_train = handle_missing(dataset)
    new_train = handle_cateogry(new_train)
    new_train = tokenize(new_train, result_tokenizer)
    X = form_input_to_model(new_train, lb_brand_name)

    return X

X_test_new = prepare_test(X_test)
y_test_new = make_y(y_test)
y_test_new = np.array(y_test_new)


# In[ ]:


print(len(X_train))
print(len(X_test))


# In[ ]:


def rmsle(y, y_pred):
      return np.sqrt(np.mean(np.power(np.log1p(y) - np.log1p(y_pred), 2)))

def evaluate_orig_price(y_test, preds):
    preds_exmpm = np.expm1(preds)
    y_test_exmpm = np.expm1(y_test)
    
    return rmsle(y_test_exmpm, preds_exmpm)

def model_train(model, X_input, y):

    X = np.hstack(X_input)
    print(X.shape)
    model.fit(X, y)

    return model


# In[ ]:


from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
import sklearn

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from mlxtend.regressor import StackingCVRegressor



# In[ ]:


print(X_train_new['name'].shape)


# In[ ]:




grid = {}
grid['alpha'] = np.arange(0, 1, 0.1)
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

search = GridSearchCV(Ridge_model, grid, scoring='neg_root_mean_squared_error', cv=cv)

X_matrix_train = (X_train_new['name'],
            X_train_new['item_desc'],
            X_train_new['cat_1'],
            X_train_new['cat_2'],
            X_train_new['cat_3'],
            X_train_new['brand_name'],            
            np.array([[i] for i in X_train_new['item_condition']]),
            np.array([[i] for i in X_train_new['shipping']])
            )

X_matrix_test = (X_test_new['name'],
            X_test_new['item_desc'],
            X_test_new['cat_1'],
            X_test_new['cat_2'],
            X_test_new['cat_3'],
            X_test_new['brand_name'],            
            np.array([[i] for i in X_test_new['item_condition']]),
            np.array([[i] for i in X_test_new['shipping']])
            )


# In[ ]:


def model_eval(model, X, y):
    X = np.hstack(X)
    preds = model.predict(X)

    # find RMSLE
    res = evaluate_orig_price(y, preds)

    print(res)
    
    print(y)
    print(preds)
    
Ridge_model = Ridge(solver='lsqr', fit_intercept=False)


# In[ ]:


Ridge_model = Ridge(alpha=0.1)
Ridge_model = model_train(Ridge_model, X_matrix_train, y_train_new)
model_eval(Ridge_model, X_matrix_train, y_train_new)
model_eval(Ridge_model, X_matrix_test, y_test_new)


# In[ ]:


Lasso_model = Lasso(alpha=0.1)
Lasso_model = model_train(Lasso_model, X_matrix_train, y_train_new)
model_eval(Lasso_model, X_matrix_train, y_train_new)
model_eval(Lasso_model, X_matrix_test, y_test_new)


# In[ ]:


lgbm_model = LGBMRegressor(n_estimators=500, learning_rate=0.2, num_leaves=250)
lgbm_model = model_train(lgbm_model, X_matrix_train, y_train_new)
model_eval(lgbm_model, X_matrix_train, y_train_new)
model_eval(lgbm_model, X_matrix_test, y_test_new)


# In[ ]:


import random

rand = random.seed(9001)

rf = RandomForestRegressor(random_state=rand, n_estimators=10)
rf = model_train(rf, X_matrix_train, y_train_new)
model_eval(rf, X_matrix_train, y_train_new)
model_eval(rf, X_matrix_test, y_test_new)


# In[ ]:


import random

rand = random.seed(9001)
ridge = Ridge(random_state=rand)
lasso = Lasso(random_state=rand)
rf = RandomForestRegressor(random_state=rand)

stack = StackingCVRegressor(regressors=(lasso, ridge),
                            meta_regressor=rf, 
                            random_state=rand,
                            use_features_in_secondary=True)

params = {'lasso__alpha': [0.1, 1.0, 2.0],
          'ridge__alpha': [0.1, 1.0, 2.0]}

grid = GridSearchCV(
    estimator=stack, 
    param_grid={
        'lasso__alpha': [x/5.0 for x in range(1, 2)],
        'ridge__alpha': [x/20.0 for x in range(1, 2)],
    }, 
    cv=2,
    refit=True
)

X = np.hstack(X_matrix_train)
grid.fit( X, y_train_new)


# In[ ]:


model_eval(grid, X_matrix_train, y_train_new)
model_eval(grid, X_matrix_test, y_test_new)


# In[ ]:


import random

rand = random.seed(9001)
ridge = Ridge(random_state=rand)
lasso = Lasso(random_state=rand)
lgbm_model = LGBMRegressor(n_estimators=500, learning_rate=0.2, num_leaves=250)


stack = StackingCVRegressor(regressors=(lasso, ridge, lgbm_model),
                            meta_regressor=rf, 
                            random_state=rand,
                            use_features_in_secondary=True)

params = {'lasso__alpha': [0.1, 1.0, 10.0],
          'ridge__alpha': [0.1, 1.0, 10.0]}

grid2 = GridSearchCV(
    estimator=stack, 
    param_grid={
        'lasso__alpha': [x/5.0 for x in range(1, 10)],
        'ridge__alpha': [x/20.0 for x in range(1, 10)],
        'meta_regressor__n_estimators': [10, 100]
    }, 
    cv=5,
    refit=True
)

X = np.hstack(X_matrix_train)
grid2.fit( X, y_train_new)


# In[ ]:




