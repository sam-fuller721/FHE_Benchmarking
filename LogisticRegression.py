import numpy as np
import pandas as pd
from Pyfhel import Pyfhel, PyCtxt
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from timeit import default_timer as timer
from HelperFunctions import *

'''
    Standard Logisic Regression
'''


def run_logistic_reg(results_dataframe: pd.DataFrame) -> pd.DataFrame:
    # check if the results dataframe has been initialized yet
    if results_dataframe.empty:
        results_dataframe = pd.DataFrame(
            columns=["Size_Data", "Size_Coefficients", "Inference_Time", "Prediction_Size", "Accuracy"])
    run_results = []

    data, target = datasets.load_iris(return_X_y=True)
    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=.5, random_state=None)
    # log size of unencrypted test data
    run_results += [data_test.nbytes]
    # preprocess data
    scaler = preprocessing.StandardScaler().fit(data_train)
    data_train = scaler.transform(data_train)
    data_test = scaler.transform(data_test)
    lin_reg = LogisticRegression()
    lin_reg.fit(data_train, target_train)
    # log the size of the trained coefficients
    run_results += [lin_reg.coef_.nbytes + lin_reg.intercept_.nbytes]
    # run inference
    start = timer()
    y_pred = lin_reg.predict(data_test)
    stop = timer()
    # log the inferencing time
    run_results += [stop - start]
    # log the prediction size
    run_results += [y_pred.nbytes]
    # log the accuracy
    run_results += [metrics.accuracy_score(target_test, y_pred)]
    # append results of the current run to the results dataframe
    results_dataframe.loc[len(results_dataframe.index)] = run_results
    return results_dataframe


'''
    AES Logisic Regression
'''


def run_logistic_reg_aes(results_dataframe: pd.DataFrame) -> pd.DataFrame:
    # check if the results dataframe has been initialized yet
    if results_dataframe.empty:
        results_dataframe = pd.DataFrame(
            columns=["Size_Data", "Size_Coefficients", "Encryption_Time_Data", "Encryption_Size_Data",
                     "Decryption_Time_Data", "Inference_Time", "Prediction_Size", "Accuracy"])
    run_results = []

    data, target = datasets.load_iris(return_X_y=True)
    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=.5, random_state=None)

    # log size of unencrypted test data
    run_results += [data_test.nbytes]

    # preprocess data
    scaler = preprocessing.StandardScaler().fit(data_train)
    data_train = scaler.transform(data_train)
    data_test = scaler.transform(data_test)
    lin_reg = LogisticRegression()
    lin_reg.fit(data_train, target_train)
    run_results += [lin_reg.coef_.nbytes + lin_reg.intercept_.nbytes]

    start = timer()
    secretKey = get_secret_key()
    # encrypt test data
    encrypted_test_data = encrypt_AES_GCM(data_test.tobytes(), secretKey)
    stop = timer()
    # log time to encrypt test data
    run_results += [stop - start]
    # log size of encrypted data
    run_results += [encrypted_test_data[0].__sizeof__()]

    # decrypt test data
    start = timer()
    decrypted_test_data = np.frombuffer(decrypt_AES_GCM(encrypted_test_data, secretKey))
    stop = timer()
    # log time to decrypt test data
    run_results += [stop - start]
    decrypted_test_data.resize((data_test.shape))
    # run inferencing
    start = timer()
    y_pred = lin_reg.predict(decrypted_test_data)
    stop = timer()
    # log the inferencing time
    run_results += [stop - start]
    # log the prediction size
    run_results += [y_pred.nbytes]
    # log the accuracy
    run_results += [metrics.accuracy_score(target_test, y_pred)]
    # append results of the current run to the results dataframe
    results_dataframe.loc[len(results_dataframe.index)] = run_results
    return results_dataframe


'''
    FHE Logisic Regression
'''


def run_logistic_reg_fhe(results_dataframe: pd.DataFrame, params=None) -> pd.DataFrame:
    # check if the results dataframe has been initialized yet
    if results_dataframe.empty:
        results_dataframe = pd.DataFrame(
            columns=["Encryption_Time_Data", "Encryption_Size_Data", "Encode_Time_Coefficients",
                     "Encode_Size_Coefficients", "Inference_Time", "Prediction_Size", "Decryption_Prediciton_Time",
                     "Accuracy"])
    run_results = []

    # load the test iris dataset
    data, target = datasets.load_iris(return_X_y=True)
    # split into train and test partitions
    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=.5, random_state=None)
    # preprocess data
    scaler = preprocessing.StandardScaler().fit(data_train)
    data_train = scaler.transform(data_train)
    data_test = scaler.transform(data_test)
    lin_reg = LogisticRegression()
    lin_reg.fit(data_train, target_train)

    # encrypt test data
    HE = Pyfhel()
    if params is not None:
        ckks_params = params
    else:
        ckks_params = {
            "scheme": "CKKS",
            "n": 2 ** 14,
            "scale": 2 ** 30,
            "qi_sizes": [60, 30, 30, 30, 60]
        }
    # time encryption process
    start = timer()
    HE.contextGen(**ckks_params)
    HE.keyGen()
    HE.relinKeyGen()
    HE.rotateKeyGen()
    data_appended = np.append(data_test, np.ones((len(data_test), 1)), -1)
    encrypted_test_data = [HE.encryptFrac(row) for row in data_appended]
    stop = timer()
    # log time to encrypt data
    run_results += [stop - start]
    # log size of encrypted data
    run_results += [get_encrypted_size(encrypted_test_data)]

    # encode coefficients from trained model, start of inferencing process
    start = timer()
    coefs = []
    for i in range(0, 3):
        coefs.append(np.append(lin_reg.coef_[i], lin_reg.intercept_[i]))
    encoded_coefs = [HE.encodeFrac(coef) for coef in coefs]
    stop = timer()
    # log time to encode coefficients
    run_results += [stop - start]
    # log size of encoded coefficients
    run_results += [get_encrypted_size(encoded_coefs)]
    # run inference
    start = timer()
    predictions = []
    for data in encrypted_test_data:
        encrypted_prediction = [
            HE.scalar_prod_plain(data, encoded_coef, in_new_ctxt=True)
            for encoded_coef in encoded_coefs
        ]
        predictions.append(encrypted_prediction)
    stop = timer()
    # log time to run inferencing
    run_results += [stop - start]
    # log prediction size
    run_results += [get_encrypted_size(predictions)]

    # decrypt predictions and check accuracy
    start = timer()
    c1_preds = []
    for prediction in predictions:
        cl = np.argmax([HE.decryptFrac(logit)[0] for logit in prediction])
        c1_preds.append(cl)
    stop = timer()
    # log prediction decryption time
    run_results += [stop - start]
    # log accuracy
    run_results += [metrics.accuracy_score(target_test, c1_preds)]
    # append results of the current run to the results dataframe
    results_dataframe.loc[len(results_dataframe.index)] = run_results
    return results_dataframe
