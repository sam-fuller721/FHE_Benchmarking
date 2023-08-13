import numpy as np
import pandas as pd
from Pyfhel import Pyfhel, PyCtxt
from timeit import default_timer as timer
from HelperFunctions import *


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


def run_mat_scalef_AES(results_dataframe: pd.DataFrame, n: int, m: int, mat_scale: int, scale_multiplier: int) -> pd.DataFrame:
    # check if the results dataframe has been initialized yet
    if results_dataframe.empty:
        results_dataframe = pd.DataFrame(columns=["Size_Data", "Encryption_Time", "Encryption_Size", "Decryption_Time"])
    run_results = []

    # Size_Data: log size of matrix a
    a = np.random.randint(mat_scale, size=(n, m))
    run_results += [a.nbytes]

    # encrypting matrix a
    start = timer()
    secretKey = get_secret_key()
    a_encrypt = encrypt_AES_GCM(a.tobytes(), secretKey)
    
    # Encryption_Time: log encryption time
    stop = timer()
    run_results += [stop - start]

    # Encryption_Size: log encryption size
    run_results += [a_encrypt[0].__sizeof__()]

    # decrypt matrix a
    start = timer()
    a_decrypt = np.frombuffer(decrypt_AES_GCM(a_encrypt, secretKey))
    a_decrypt.resize((a.shape))

    # Decryption_Time: log decryption time
    stop = timer()
    run_results += [stop - start]

    res = a_decrypt * scale_multiplier
    results_dataframe.loc[len(results_dataframe.index)] = run_results
    return results_dataframe


def run_mat_scalei_AES(results_dataframe: pd.DataFrame, n: int, m: int, mat_scale: int, scale_multiplier: int) -> pd.DataFrame:
    # check if the results dataframe has been initialized yet
    if results_dataframe.empty:
        results_dataframe = pd.DataFrame(columns=["Size_Data", "Encryption_Time", "Encryption_Size", "Decryption_Time"])
    run_results = []

    # Size_Data: log size of matrix a
    a = np.random.randint(mat_scale, size=(n, m))
    run_results += [a.nbytes]

    # encrypting matrix a
    start = timer()
    secretKey = get_secret_key()
    a_encrypt = encrypt_AES_GCM(a.tobytes(), secretKey)
    
    # Encryption_Time: log encryption time
    stop = timer()
    run_results += [stop - start]

    # Encryption_Size: log encryption size
    run_results += [a_encrypt[0].__sizeof__()]

    # decrypt matrix a
    start = timer()
    a_decrypt = np.frombuffer(decrypt_AES_GCM(a_encrypt, secretKey), dtype=int)
    a_decrypt.resize((a.shape))

    # Decryption_Time: log decryption time
    stop = timer()
    run_results += [stop - start]

    res = a_decrypt * scale_multiplier
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
