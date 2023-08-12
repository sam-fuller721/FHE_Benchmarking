import numpy as np
import pandas as pd
from Pyfhel import Pyfhel, PyCtxt
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from timeit import default_timer as timer
import argparse 
import json
import sys
import os
from tqdm import tqdm 
from Crypto.Cipher import AES
from Crypto import Random


def get_encrypted_size(encrypted_data) -> int:
    sum = 0
    for e in encrypted_data: 
        sum += e.__sizeof__()
    return sum  

def get_encrypted_size_mat(encrypted_data) -> int: 
    sum = 0
    for row in encrypted_data: 
        sum += get_encrypted_size(row) 
    return sum

def percent_error_matrix(mat_ref, mat_calculated) -> float: 
    mat_diff = mat_ref - mat_calculated 
    return np.average(mat_diff)

'''
    Helper functions for AES256 Encryption, GCM mode
    Source: https://cryptobook.nakov.com/symmetric-key-ciphers/aes-encrypt-decrypt-examplesfrom Crypto.Cipher import AES

'''
def get_secret_key():
    return Random.get_random_bytes(32) # 256-bit random encryption key

def encrypt_AES_GCM(msg, secretKey):
    aesCipher = AES.new(secretKey, AES.MODE_GCM)
    ciphertext, authTag = aesCipher.encrypt_and_digest(msg)
    return (ciphertext, aesCipher.nonce, authTag)

def decrypt_AES_GCM(encryptedMsg, secretKey):
    (ciphertext, nonce, authTag) = encryptedMsg
    aesCipher = AES.new(secretKey, AES.MODE_GCM, nonce)
    plaintext = aesCipher.decrypt_and_verify(ciphertext, authTag)
    return plaintext


def run_mat_scalef(results_dataframe: pd.DataFrame, n: int, m: int, mat_scale: int, scale_multiplier: int) -> pd.DataFrame:
    # check if the results dataframe has been initialized yet
    if results_dataframe.empty:
        results_dataframe = pd.DataFrame(columns=["Size_Data", "Processing_Time", "Size_Results"])
    run_results = []
    a = np.random.random((n, m)) * mat_scale
    run_results += [a.nbytes]
    start = timer()
    res = a * scale_multiplier
    stop = timer()
    run_results += [stop - start]
    run_results += [res.nbytes]
    results_dataframe.loc[len(results_dataframe.index)] = run_results
    return results_dataframe


def run_mat_scalei(results_dataframe: pd.DataFrame, n: int, m: int, mat_scale: int, scale_multiplier: int) -> pd.DataFrame:
    # check if the results dataframe has been initialized yet
    if results_dataframe.empty:
        results_dataframe = pd.DataFrame(columns=["Size_Data", "Processing_Time", "Size_Results"])
    run_results = []
    a = np.random.randint(mat_scale, size=(n, m))
    run_results += [a.nbytes]
    start = timer()
    res = a * scale_multiplier
    stop = timer()
    run_results += [stop - start]
    run_results += [res.nbytes]
    results_dataframe.loc[len(results_dataframe.index)] = run_results
    return results_dataframe


def run_mat_scalef_FHE(results_dataframe: pd.DataFrame, n: int, m: int, mat_scale: int, scale_multiplier: int, params=None) -> pd.DataFrame:
    # check if the results dataframe has been initialized yet
    if results_dataframe.empty:
        results_dataframe = pd.DataFrame(
            columns=["Encryption_Time_Data", "Encryption_Size_Data", "Encode_Time_Scalar", "Encode_Size_Scalar",
                     "Processing_Time", "Encryption_Size_Results", "Decryption_Results_Time", "Accuracy"])
    run_results = []
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
    a_mat = np.random.random((n, m)) * mat_scale
    start = timer()
    HE.contextGen(**ckks_params)
    HE.keyGen()
    HE.relinKeyGen()
    HE.rotateKeyGen()
    # encrypt matrix
    a_enc = [HE.encryptFrac(np.array([elem])) for elem in a_mat.flatten("C")]
    stop = timer()
    # log time to encrypt matrix
    run_results += [stop - start]
    # log size of encrypted matrix
    run_results += [get_encrypted_size(a_enc)]
    start = timer()
    scale_enc = HE.encodeFrac(np.array([scale_multiplier], dtype=np.float64))
    stop = timer()
    # log time to encode the scalar multiplier
    run_results += [stop - start]
    # log the size of the scalar multiplier
    run_results += [scale_enc.__sizeof__()]
    start = timer()
    res = [elem * scale_enc for elem in a_enc]
    stop = timer()
    # log the time to scale the encrypted matrix
    run_results += [stop - start]
    # log the resulting size of the encrypted matrix
    run_results += [get_encrypted_size(res)]
    start = timer()
    res_decrypted = []
    for row in range(n):
        temp = []
        for col in range(m):
            temp.append(HE.decryptFrac(res[row*m + col])[0])
        res_decrypted.append(temp)
    stop = timer()
    run_results += [stop - start]
    run_results += [percent_error_matrix(a_mat * scale_multiplier, np.array(res_decrypted))]
    results_dataframe.loc[len(results_dataframe.index)] = run_results
    return results_dataframe


def run_mat_scalei_FHE(results_dataframe: pd.DataFrame, n: int, m: int, mat_scale: int, scale_multiplier: int, params=None) -> pd.DataFrame:
    # check if the results dataframe has been initialized yet
    if results_dataframe.empty:
        results_dataframe = pd.DataFrame(
            columns=["Encryption_Time_Data", "Encryption_Size_Data", "Encode_Time_Scalar", "Encode_Size_Scalar",
                     "Processing_Time", "Encryption_Size_Results", "Decryption_Results_Time", "Accuracy"])
    run_results = []
    HE = Pyfhel()
    if params is not None:
        bfv_params = params
    else:
        bfv_params = {
            'scheme': 'BFV',
            'n': 2 ** 13,
            't': 65537,
            't_bits': 20,
            'sec': 128,
        }
    a_mat = np.random.randint(mat_scale, size=(n, m))
    start = timer()
    HE.contextGen(**bfv_params)
    HE.keyGen()
    HE.rotateKeyGen()
    HE.relinKeyGen()
    # encrypt matrix
    a_enc = [HE.encryptInt(np.array([elem])) for elem in a_mat.flatten("C")]
    stop = timer()
    # log time to encrypt matrix
    run_results += [stop - start]
    # log size of encrypted matrix
    run_results += [get_encrypted_size(a_enc)]
    start = timer()
    scale_enc = HE.encodeInt(np.array([scale_multiplier], dtype=np.int64))
    stop = timer()
    # log time to encode the scalar multiplier
    run_results += [stop - start]
    # log the size of the scalar multiplier
    run_results += [scale_enc.__sizeof__()]
    start = timer()
    res = [elem * scale_enc for elem in a_enc]
    stop = timer()
    # log the time to scale the encrypted matrix
    run_results += [stop - start]
    # log the resulting size of the encrypted matrix
    run_results += [get_encrypted_size(res)]
    start = timer()
    res_decrypted = []
    for row in range(n):
        temp = []
        for col in range(m):
            temp.append(HE.decryptInt(res[row * m + col])[0])
        res_decrypted.append(temp)
    stop = timer()
    run_results += [stop - start]
    run_results += [percent_error_matrix(a_mat * scale_multiplier, np.array(res_decrypted))]
    results_dataframe.loc[len(results_dataframe.index)] = run_results
    return results_dataframe


'''
    Standard Matrix Multiplication - Float
'''
def run_mat_mulf(results_dataframe: pd.DataFrame, n: int, m: int, scale: int) -> pd.DataFrame:
    # check if the results dataframe has been initialized yet 
    if results_dataframe.empty: 
        results_dataframe = pd.DataFrame(columns=["Size_Data", "Processing_Time", "Size_Results"])
    run_results = []
    a = np.random.random((n, m)) * scale
    b = np.random.random((m, n)) * scale
    # log size of both matrices
    run_results += [a.nbytes + b.nbytes]
    start = timer()
    res = a @ b
    stop = timer()
    # log the processing time 
    run_results += [stop - start]
    # log the size of the resulting matrix 
    run_results += [res.nbytes]
    # append results of the current run to the results dataframe
    results_dataframe.loc[len(results_dataframe.index)] = run_results
    return results_dataframe


'''
    Standard Matrix Multiplication - Integer
'''
def run_mat_muli(results_dataframe: pd.DataFrame, n: int, m: int, scale: int) -> pd.DataFrame:
    # check if the results dataframe has been initialized yet 
    if results_dataframe.empty: 
        results_dataframe = pd.DataFrame(columns=["Size_Data", "Processing_Time", "Size_Results"])
    run_results = []
    a = np.random.randint(scale, size=(n, m))
    b = np.random.randint(scale, size=(m, n))
    # log size of both matrices
    run_results += [a.nbytes + b.nbytes]
    start = timer()
    res = a @ b
    stop = timer()
    # log the processing time 
    run_results += [stop - start]
    # log the size of the resulting matrix 
    run_results += [res.nbytes]
    # append results of the current run to the results dataframe
    results_dataframe.loc[len(results_dataframe.index)] = run_results
    return results_dataframe


'''
    AES Matrix Multiplication - Float
'''
def run_mat_mulf_aes(results_dataframe: pd.DataFrame, n: int, m: int, scale: int) -> pd.DataFrame:
    # check if the results dataframe has been initialized yet 
    if results_dataframe.empty: 
        results_dataframe = pd.DataFrame(columns=["Size_Data", "Processing_Time", "Size_Results", "Accuracy"])
    run_results = []

    a = np.random.random((n, m)) * scale
    b = np.random.random((m, n)) * scale
    
    # log size of both matrices
    run_results += [a.nbytes + b.nbytes]

    start = timer()
    secretKey = get_secret_key()
    
    # encrypting matrices, a and b
    a_encrypt = encrypt_AES_GCM(a.tobytes(), secretKey)
    b_encrypt = encrypt_AES_GCM(b.tobytes(), secretKey)
    
    # decrypting encrypted matrices
    a_decrypt = np.frombuffer(decrypt_AES_GCM(a_encrypt, secretKey))
    a_decrypt.resize((a.shape)) 
    b_decrypt = np.frombuffer(decrypt_AES_GCM(b_encrypt, secretKey))
    b_decrypt.resize((b.shape))

    res = a_decrypt @ b_decrypt
    stop = timer()

    # log the processing time 
    run_results += [stop - start]

    # log the size of the resulting matrix 
    run_results += [get_encrypted_size_mat(a_encrypt) + get_encrypted_size_mat(b_encrypt) + a_decrypt.nbytes + b_decrypt.nbytes]

    # log the percent error from the gold standard numpy matrix multiply 
    run_results += [percent_error_matrix(a @ b, res)]

    # append results of the current run to the results dataframe
    results_dataframe.loc[len(results_dataframe.index)] = run_results

    return results_dataframe


'''
    AES Matrix Multiplication - Integer
'''
def run_mat_muli_aes(results_dataframe: pd.DataFrame, n: int, m: int, scale: int) -> pd.DataFrame:
    # check if the results dataframe has been initialized yet
    if results_dataframe.empty: 
        results_dataframe = pd.DataFrame(columns=["Size_Data", "Processing_Time", "Size_Results", "Accuracy"])
    run_results = []

    a = np.random.randint((n, m)) * scale
    b = np.random.randint((m, n)) * scale
    
    # log size of both matrices
    run_results += [a.nbytes + b.nbytes]

    start = timer()
    secretKey = get_secret_key()
    
    # encrypting matrices, a and b
    a_encrypt = encrypt_AES_GCM(a.tobytes(), secretKey)
    b_encrypt = encrypt_AES_GCM(b.tobytes(), secretKey)
    
    # decrypting encrypted matrices
    a_decrypt = np.frombuffer(decrypt_AES_GCM(a_encrypt, secretKey), dtype=int)
    a_decrypt.resize((a.shape)) 
    b_decrypt = np.frombuffer(decrypt_AES_GCM(b_encrypt, secretKey), dtype=int)
    b_decrypt.resize((b.shape))

    res = a_decrypt @ b_decrypt
    stop = timer()

    # log the processing time 
    run_results += [stop - start]

    # log the size of the resulting matrix 
    run_results += [get_encrypted_size_mat(a_encrypt) + get_encrypted_size_mat(b_encrypt) + a_decrypt.nbytes + b_decrypt.nbytes]

    # log the percent error from the gold standard numpy matrix multiply 
    run_results += [percent_error_matrix(a @ b, res)]

    # append results of the current run to the results dataframe
    results_dataframe.loc[len(results_dataframe.index)] = run_results

    return results_dataframe


'''
    FHE Matrix Multiplication - Integer
'''
def run_mat_muli_fhe(results_dataframe: pd.DataFrame, n: int, m: int, scale: int, params=None) -> pd.DataFrame:
    # check if the results dataframe has been initialized yet 
    if results_dataframe.empty: 
        results_dataframe = pd.DataFrame(columns=["Encryption_Time_Data", "Encryption_Size_Data", "Processing_Time", "Encryption_Size_Results", "Decryption_Results_Time", "Accuracy"])
    run_results = []
    HE = Pyfhel()
    if params is not None: 
        bfv_params = params
    else:
        bfv_params = {
            'scheme': 'BFV',
            'n': 2 ** 13,
            't': 65537,
            't_bits': 20,
            'sec': 128,
        }
    start = timer()
    HE.contextGen(**bfv_params)
    HE.keyGen()
    HE.rotateKeyGen()
    HE.relinKeyGen()
    a_mat = np.random.randint(scale, size=(n, m))
    b_mat = np.random.randint(scale, size=(m, n))
    a_enc = [HE.encryptInt(np.array(row)) for row in a_mat]
    b_enc = [HE.encryptInt(np.array(col)) for col in b_mat.T] 
    stop = timer()
    # log time to encrypt both matrices 
    run_results += [stop - start]
    # log encrypted size of both matrices 
    run_results += [get_encrypted_size(a_enc) + get_encrypted_size(b_enc)]

    start = timer()
    res = []
    for a_row in a_enc:
        sub_res = []
        for b_col in b_enc:
            sub_res.append(HE.scalar_prod(a_row, b_col, in_new_ctxt=True))
        res.append(sub_res)
    stop = timer()
    # log processing time 
    run_results += [stop - start]
    # log size of resulting matrix 
    run_results += [get_encrypted_size_mat(res)]
    # decrypt matrix 
    res_decrypt = []
    start = timer()
    for row in res:
        temp = []
        for elem in row: 
            temp.append(HE.decryptInt(elem)[0])
        res_decrypt.append(temp)
    stop = timer()
    res_np = np.array(res_decrypt)
    # log time to decrypt results 
    run_results += [stop - start]
    # log the percent error from the gold standard numpy matrix multiply 
    run_results += [percent_error_matrix(a_mat @ b_mat, res_np)]
    # append results of the current run to the results dataframe
    results_dataframe.loc[len(results_dataframe.index)] = run_results
    return results_dataframe


'''
    FHE Matrix Multiplication - Float
'''
def run_mat_mulf_fhe(results_dataframe: pd.DataFrame, n: int, m: int, scale: int, params=None) -> pd.DataFrame:
    # check if the results dataframe has been initialized yet 
    if results_dataframe.empty: 
        results_dataframe = pd.DataFrame(columns=["Encryption_Time_Data", "Encryption_Size_Data", "Processing_Time", "Encryption_Size_Results", "Decryption_Results_Time", "Accuracy"])
    run_results = []

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

    start = timer()
    HE.contextGen(**ckks_params)
    HE.keyGen()
    HE.relinKeyGen()
    HE.rotateKeyGen()
    a_mat = np.random.random((n, m)) * scale
    b_mat = (np.random.random((m, n)) * scale)
    a_enc = [HE.encryptFrac(np.array(row)) for row in a_mat]
    b_enc = [HE.encryptFrac(np.array(col)) for col in b_mat.T]
    stop = timer() 
    # log time to encrypt both matrices 
    run_results += [stop - start]
    # log encrypted size of both matrices 
    run_results += [get_encrypted_size(a_enc) + get_encrypted_size(b_enc)]

    start = timer()
    res = []
    for a_row in a_enc:
        sub_res = []
        for b_col in b_enc:
            sub_res.append(HE.scalar_prod(a_row, b_col, in_new_ctxt=True))
        res.append(sub_res)
    stop = timer()
    # log processing time 
    run_results += [stop - start]
    # log size of resulting matrix 
    run_results += [get_encrypted_size_mat(res)]
    # decrypt the matrix
    res_decrypt = []
    start = timer()
    for row in res:
        temp = []
        for elem in row: 
            temp.append(HE.decryptFrac(elem)[0])
        res_decrypt.append(temp)
    stop = timer()
    res_np = np.array(res_decrypt)
    # log time to decrypt results 
    run_results += [stop - start]
    # log the percent error from the gold standard numpy matrix multiply 
    run_results += [percent_error_matrix(a_mat @ b_mat, res_np)]
    # append results of the current run to the results dataframe
    results_dataframe.loc[len(results_dataframe.index)] = run_results
    return results_dataframe


'''
    Standard Logisic Regression
'''
def run_logistic_reg(results_dataframe: pd.DataFrame) -> pd.DataFrame:
    # check if the results dataframe has been initialized yet 
    if results_dataframe.empty: 
        results_dataframe = pd.DataFrame(columns=["Size_Data", "Size_Coefficients", "Inference_Time", "Prediction_Size", "Accuracy"])
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
    run_results += [lin_reg.coef_.nbytes]
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
        results_dataframe = pd.DataFrame(columns=["Size_Data", "Size_Coefficients", "Inference_Time", "Prediction_Size", "Accuracy"])
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
    secretKey = get_secret_key()

    # run inference
    start = timer()
    
    # encrypt test data, train data, target train
    encrypted_test_data = encrypt_AES_GCM(data_test.tobytes(), secretKey)
    decrypted_test_data = np.frombuffer(decrypt_AES_GCM(encrypted_test_data, secretKey))
    decrypted_test_data.resize((data_test.shape)) 

    encrypted_train_data = encrypt_AES_GCM(data_train.tobytes(), secretKey)
    decrypted_train_data = np.frombuffer(decrypt_AES_GCM(encrypted_train_data, secretKey))
    decrypted_train_data.resize((data_train.shape))

    encrypted_target_train = encrypt_AES_GCM(target_train.tobytes(), secretKey)
    decrypted_target_train = np.frombuffer(decrypt_AES_GCM(encrypted_target_train, secretKey), dtype=int)
    decrypted_target_train.resize((target_train.shape))    

    lin_reg.fit(decrypted_train_data, decrypted_target_train)

    # log the size of the trained coefficients 
    run_results += [lin_reg.coef_.nbytes]

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
        results_dataframe = pd.DataFrame(columns=["Encryption_Time_Data", "Encryption_Size_Data", "Encode_Time_Coefficients", "Encode_Size_Coefficients", "Inference_Time", "Prediction_Size", "Decryption_Prediciton_Time", "Accuracy"])
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


def main(args):
    if not os.path.exists("test_results"):
        os.mkdir("test_results")
    # create dict for storing a mapping of test types to test functions 
    test_functions = {
        "log_reg_float_FHE": run_logistic_reg_fhe,
        "log_reg_float_AES": run_logistic_reg_aes,
        "log_reg_float_NONE": run_logistic_reg,
        "mat_mul_float_FHE": run_mat_mulf_fhe,
        "mat_mul_int_FHE": run_mat_muli_fhe,
        "mat_mul_float_AES": run_mat_mulf_aes,
        "mat_mul_int_AES": run_mat_muli_aes,
        "mat_mul_float_NONE": run_mat_mulf,
        "mat_mul_int_NONE":  run_mat_muli,
        "mat_scale_float_NONE": run_mat_scalef,
        "mat_scale_int_NONE": run_mat_scalei,
        "mat_scale_float_FHE": run_mat_scalef_FHE,
        "mat_scale_int_FHE": run_mat_scalei_FHE
    }
    # parse the test file JSON 
    if args.file_input:
        test_desc = json.load(open(args.file_input)) 
        test_file_path = "test_results/"
        # iterate over list of tests in test file
        with open(test_file_path + test_desc["out_file"], "w") as logger:    # log outputs to the specified out file
            cnt = 0
            for test in test_desc["tests"]:
                logger.write(f'Running Test: {cnt}, {test["type"]}\n')
                # look-up the correct test function according to test type 
                func =  test_functions[test["type"]]
                # populate the test function arguments 
                args = [pd.DataFrame()]     # start with an empty dataframe to store test results 
                # load matrices size and scale if they are set in the test
                if "mat_size" in test: 
                    args += [test["mat_size"][0], test["mat_size"][1]]
                if "scale" in test: 
                    args += [test["scale"]]
                if "scale_multiplier" in test:
                    args += [test["scale_multiplier"]]
                # if the file defines pyfhel params, use them 
                if "pyfhel_params" in test:
                    args += [test["pyfhel_params"]]
                # run the test
                time = 0.0
                runs = test["runs"]
                logger.write(f'Using Args: {args[1:]} ')
                for _ in tqdm(range(runs), desc=f'Running Test {cnt}: {test["type"]}', ascii=True):
                    args[0] = func(*args)
                logger.write(f'Finished Running Test: {cnt}\n')
                # log specifc results in a csv for further analysis 
                args[0].to_csv(f'{test_file_path}/{test["type"]}_results.csv')
                cnt += 1 
    else:
        print("Only file input is supported")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Bench Tool for comparing FHE, AES, and Non-encrypted operations', fromfile_prefix_chars='@')
    parser.add_argument('--file_input', '-f', help='Test Description File: takes a string to a test description file, expects a JSON file describing tests to be run')
    parser.add_argument('--test_type', '-t', help='Test Type: takes a string denoting the type of test to perform (mat_mul, mat_scale, log_reg, add, sub, mult, div)')
    parser.add_argument('--encryption', '-e', help='Encryption Type: takes a string denoting the type of encryption to use (FHE, AES, NONE)')
    parser.add_argument('--data_type', '-d', help='Data Type: takes a string denoting the type of data used during the test (float, int)')
    parser.add_argument('--runs', '-r', help='Test Runs: takes an integer number of iterations to run the test', default=10)
    parser.add_argument('--scale', '-s', help='Data Scale: Scale factor applied to randomly generated data, not applicable to log_reg test', default=1)
    parser.add_argument('--mat_size', '-m', help='Matrix Size: takes an integer tuple (n, m) denoting the size of the matrix to be tested with. Only applicable to mat_mul and mat_scale tests', default=(5, 5))
    parser.add_argument('--out_file', '-o', help='Outfile Name: takes a string denoting the name of the outfile to store test results in', default='results.txt')
    main(parser.parse_args())
