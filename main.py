import pandas as pd
import numpy as np
import gc
import tensorflow as tf
import xgboost as xgb
import lightgbm as lgb
import collections
import operator

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score

from keras import backend as K
from keras import optimizers
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Bidirectional
from keras.layers import Convolution1D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import GRU
from keras.layers import LSTM
from keras.layers import GlobalMaxPool1D
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import SpatialDropout1D
from keras.layers import concatenate
from keras.models import Model
from keras.preprocessing import sequence
from keras.preprocessing import text


cat_features = [
    'Day of Week',
    'Day of Month',
    'Month of Year']

cont_features = [
    'Open', 'High',
    'Low', 'Volume',
    'Adj Close']

target_feature = ['PChange']

seed = 2019
win_size = 25

batch_size = 32
epochs = 16

np.random.seed(seed)
tf.set_random_seed(seed)


def build_model_vanilla(
    x_input_shape,
    output_shape,
    dropout_rate=0,
    kernel_regularizer=0,
    activity_regularizer=0,
    bias_regularizer=0,
    **kwargs):

    x_input = Input(x_input_shape, name='x_input')
    x = Flatten()(x_input)
    x = Dense(
        32,
        activation='relu',
        kernel_regularizer=regularizers.l2(kernel_regularizer),
        bias_regularizer=regularizers.l2(bias_regularizer))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(
        16,
        activation='relu',
        kernel_regularizer=regularizers.l2(kernel_regularizer),
        bias_regularizer=regularizers.l2(bias_regularizer))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(
        8,
        activation='relu',
        kernel_regularizer=regularizers.l2(kernel_regularizer),
        bias_regularizer=regularizers.l2(bias_regularizer))(x)
    x = Dropout(dropout_rate)(x)

    output = Dense(output_shape, activation='sigmoid', name='output')(x)

    model = Model(inputs=[x_input], outputs=[output])
    model.compile(optimizer=optimizers.Adam(0.0005, decay=1e-6),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model
#end def


def build_model_gru(
    x_input_shape,
    output_shape,
    dropout_rate=0,
    kernel_regularizer=0,
    activity_regularizer=0,
    bias_regularizer=0,
    **kwargs):

    x_input = Input(x_input_shape, name='x_input')
    x = Bidirectional(GRU(16, return_sequences=True))(x_input)
    x = Convolution1D(8, 3, activation="relu")(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(
        8,
        activation='relu',
        kernel_regularizer=regularizers.l2(kernel_regularizer),
        bias_regularizer=regularizers.l2(bias_regularizer))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(
        4,
        activation='relu',
        kernel_regularizer=regularizers.l2(kernel_regularizer),
        bias_regularizer=regularizers.l2(bias_regularizer))(x)
    x = Dropout(dropout_rate)(x)

    output = Dense(output_shape, activation='sigmoid', name='output')(x)

    model = Model(inputs=[x_input], outputs=[output])
    model.compile(optimizer=optimizers.Adam(0.0005, decay=1e-6),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model
#end def


def build_model_lstm(
    x_input_shape,
    output_shape,
    dropout_rate=0,
    kernel_regularizer=0,
    activity_regularizer=0,
    bias_regularizer=0,
    **kwargs):

    x_input = Input(x_input_shape, name='x_input')
    x = Bidirectional(LSTM(16, return_sequences=True))(x_input)
    x = Convolution1D(8, 3, activation="relu")(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(
        8,
        activation='relu',
        kernel_regularizer=regularizers.l2(kernel_regularizer),
        bias_regularizer=regularizers.l2(bias_regularizer))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(
        4,
        activation='relu',
        kernel_regularizer=regularizers.l2(kernel_regularizer),
        bias_regularizer=regularizers.l2(bias_regularizer))(x)
    x = Dropout(dropout_rate)(x)

    output = Dense(output_shape, activation='sigmoid', name='output')(x)

    model = Model(inputs=[x_input], outputs=[output])
    model.compile(optimizer=optimizers.Adam(0.0005, decay=1e-6),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model
#end def


def main():
    df = pd.read_csv('processed_data.csv')

    # encode categorical features
    cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    X_cat_raw = cat_encoder.fit_transform(df[cat_features].values)

    X_cont_raw = df[cont_features].values

    X_raw = np.hstack([X_cont_raw, X_cat_raw])
    Y_raw = df[target_feature].values

    ###### Prepare Data
    X = list()
    Y = list()
    for i in range(Y_raw.shape[0]):
        if i < win_size:
            continue

        _x = [X_raw[i-k] for k in range(win_size, 0, -1)]

        X.append(_x)
        Y.append(Y_raw[i][0])
    #end for

    X = np.asarray(X)
    Y = np.asarray(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)

    _class_weight = compute_class_weight('balanced', np.asarray([0, 1]), Y_train)
    class_weight = {i: weight for i, weight in enumerate(_class_weight)}

    ###### Prepare Callback
    callbacks_list = []
    early_stopping = dict(monitor='val_loss', patience=1, min_delta=0.001, verbose=1)
    model_checkpoint = dict(filepath='./weights/{val_loss:.5f}_{epoch:04d}.weights.h5',
                            save_best_only=True,
                            save_weights_only=True,
                            mode='auto',
                            period=1,
                            verbose=1)

    earlystop = EarlyStopping(**early_stopping)
    callbacks_list.append(earlystop)

    checkpoint = ModelCheckpoint(**model_checkpoint)
    callbacks_list.append(checkpoint)

    ###### Vanilla Model
    tf_session = tf.Session()
    K.set_session(tf_session)
    K.get_session().run(tf.tables_initializer())

    init_op = tf.global_variables_initializer()
    tf_session.run(init_op)

    vanilla_model = build_model_vanilla(
        x_input_shape=X_train.shape[1:],
        output_shape=1)

    vanilla_model.fit(
        x=X_train,
        y=Y_train,
        batch_size=batch_size,
        verbose=1,
        epochs=epochs,
        validation_split=0.1,
        callbacks=callbacks_list,
        class_weight=class_weight)

    Y_vanilla_pred = vanilla_model.predict(X_test)
    Y_vanilla_pred = [1 if i >= 0.5 else 0 for i in Y_vanilla_pred]

    print("="*10 + "Classification Report for Vanilla:" + "="*10)
    print(classification_report(Y_test, Y_vanilla_pred))

    print("="*10 + "Accuracy Report for Vanilla:" + "="*10)
    print(accuracy_score(Y_test, Y_vanilla_pred))
    print()

    ###### GRU Model
    tf_session = tf.Session()
    K.set_session(tf_session)
    K.get_session().run(tf.tables_initializer())

    init_op = tf.global_variables_initializer()
    tf_session.run(init_op)

    gru_model = build_model_gru(
        x_input_shape=X_train.shape[1:],
        output_shape=1)

    gru_model.fit(
        x=X_train,
        y=Y_train,
        batch_size=batch_size,
        verbose=1,
        epochs=epochs,
        validation_split=0.1,
        callbacks=callbacks_list,
        class_weight=class_weight)

    Y_gru_pred = gru_model.predict(X_test)
    Y_gru_pred = [1 if i >= 0.5 else 0 for i in Y_gru_pred]

    print("="*10 + "Classification Report for GRU:" + "="*10)
    print(classification_report(Y_test, Y_gru_pred))

    print("="*10 + "Accuracy Report for GRU:" + "="*10)
    print(accuracy_score(Y_test, Y_gru_pred))

    print()

    ###### LSTM Model
    tf_session = tf.Session()
    K.set_session(tf_session)
    K.get_session().run(tf.tables_initializer())

    init_op = tf.global_variables_initializer()
    tf_session.run(init_op)

    lstm_model = build_model_lstm(
        x_input_shape=X_train.shape[1:],
        output_shape=1)

    lstm_model.fit(
        x=X_train,
        y=Y_train,
        batch_size=batch_size,
        verbose=1,
        epochs=epochs,
        validation_split=0.1,
        callbacks=callbacks_list,
        class_weight=class_weight)

    Y_lstm_pred = lstm_model.predict(X_test)
    Y_lstm_pred = [1 if i >= 0.5 else 0 for i in Y_lstm_pred]

    print("="*10 + "Classification Report for LSTM:" + "="*10)
    print(classification_report(Y_test, Y_lstm_pred))

    print("="*10 + "Accuracy Report for LSTM:" + "="*10)
    print(accuracy_score(Y_test, Y_lstm_pred))

    print()

    ###### XGBoost
    X_train_stacked = X_train.reshape(*X_train.shape[:1], -1)
    X_test_stacked = X_test.reshape(*X_test.shape[:1], -1)

    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=8, learning_rate=0.1, subsample=0.5)
    xgb_model = xgb_model.fit(X_train_stacked, Y_train)
    Y_xgb_pred = xgb_model.predict(X_test_stacked)

    print("="*10 + "Classification Report for XGBoost:" + "="*10)
    print(classification_report(Y_test, Y_xgb_pred))

    print("="*10 + "Accuracy Report for XGBoost:" + "="*10)
    print(accuracy_score(Y_test, Y_xgb_pred))

    print()

    ###### LGBM
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=seed)
    X_train_stacked = X_train.reshape(*X_train.shape[:1], -1)
    X_val_stacked = X_val.reshape(*X_val.shape[:1], -1)
    X_test_stacked = X_test.reshape(*X_test.shape[:1], -1)

    lgb_train = lgb.Dataset(X_train_stacked, Y_train)
    lgb_test = lgb.Dataset(X_val_stacked, Y_val, reference=lgb_train)

    params = {
        'task': 'train',
        'objective': 'binary',
        'metric': 'binary_error',
        'verbose': 1}

    lgbm_model = lgb.train(
        params,
        lgb_train,
        num_boost_round=50,
        early_stopping_rounds=10,
        valid_sets=[lgb_train, lgb_test],
        valid_names=['train', 'eval'])

    Y_lgbm_pred = lgbm_model.predict(X_test_stacked)
    Y_lgbm_pred = [1 if i >= 0.53 else 0 for i in Y_lgbm_pred]

    print("="*10 + "Classification Report for LGBM:" + "="*10)
    print(classification_report(Y_test, Y_lgbm_pred))

    print("="*10 + "Accuracy Report for LGBM:" + "="*10)
    print(accuracy_score(Y_test, Y_lgbm_pred))

    print()

    ######### Ensemble Learning
    test_size = Y_test.shape[0]

    Y_pred = [1 if sum([Y_gru_pred[i], Y_lstm_pred[i], Y_xgb_pred[i]]) > 1 else 0 for i in range(test_size)]

    print("="*10 + "Classification Report for Ensemble:" + "="*10)
    print(classification_report(Y_test, Y_pred))

    print("="*10 + "Accuracy Report for Ensemble:" + "="*10)
    print(accuracy_score(Y_test, Y_pred))

    # Y_pred = [1 for i in range(test_size)]
    # print("="*10 + "Classification Report for Ensemble:" + "="*10)
    # print(classification_report(Y_test, Y_pred))

    # print("="*10 + "Accuracy Report for Ensemble:" + "="*10)
    # print(accuracy_score(Y_test, Y_pred))

    # # Performance on full dataset
    # test_size = Y.shape[0]

    # Y_vanilla_pred = vanilla_model.predict(X)
    # Y_vanilla_pred = [1 if i >= 0.5 else 0 for i in Y_vanilla_pred]

    # Y_gru_pred = gru_model.predict(X)
    # Y_gru_pred = [1 if i >= 0.5 else 0 for i in Y_gru_pred]

    # Y_lstm_pred = lstm_model.predict(X)
    # Y_lstm_pred = [1 if i >= 0.5 else 0 for i in Y_lstm_pred]

    # X_stacked = X.reshape(*X.shape[:1], -1)
    # Y_xgb_pred = xgb_model.predict(X_stacked)

    # Y_pred = [1 if sum([Y_gru_pred[i], Y_lstm_pred[i], Y_xgb_pred[i]]) > 1 else 0 for i in range(test_size)]

    # print()
    # print("="*10 + "Full Classification Report for Ensemble:" + "="*10)
    # print(classification_report(Y, Y_pred))

    # print("="*10 + "Full Accuracy Report for Ensemble:" + "="*10)
    # print(accuracy_score(Y, Y_pred))
#end def

if __name__ == '__main__': main()

