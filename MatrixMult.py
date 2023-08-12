import numpy as np
import pandas as pd
from Pyfhel import Pyfhel, PyCtxt
from timeit import default_timer as timer
from HelperFunctions import *

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
        results_dataframe = pd.DataFrame(columns=["Size_Data", "Encryption_Time", "Encryption_Size", "Decryption_Time"])
    run_results = []

    a = np.random.random((n, m)) * scale
    b = np.random.random((m, n)) * scale
    
    # Size_Data: log size of both matrices
    run_results += [a.nbytes + b.nbytes]

    # encrypting matrices, a and b
    start = timer()
    secretKey = get_secret_key()
    
    a_encrypt = encrypt_AES_GCM(a.tobytes(), secretKey)
    b_encrypt = encrypt_AES_GCM(b.tobytes(), secretKey)
    
    # Encryption_Time: log encryption time
    stop = timer()
    run_results += [stop - start]

    # Encryption_Size: log size of encrypted matrices
    run_results += [get_encrypted_size_mat(a_encrypt) + get_encrypted_size_mat(b_encrypt)]

    # decrypting encrypted matrices
    start = timer()
    a_decrypt = np.frombuffer(decrypt_AES_GCM(a_encrypt, secretKey))
    a_decrypt.resize((a.shape)) 
    b_decrypt = np.frombuffer(decrypt_AES_GCM(b_encrypt, secretKey))
    b_decrypt.resize((b.shape))

    res = a_decrypt @ b_decrypt

    # Decryption_Time: log decryption time
    stop = timer()
    run_results += [stop - start]

    # append results of the current run to the results dataframe
    results_dataframe.loc[len(results_dataframe.index)] = run_results

    return results_dataframe


'''
    AES Matrix Multiplication - Integer
'''


def run_mat_muli_aes(results_dataframe: pd.DataFrame, n: int, m: int, scale: int) -> pd.DataFrame:
    # check if the results dataframe has been initialized yet 
    if results_dataframe.empty: 
        results_dataframe = pd.DataFrame(columns=["Size_Data", "Encryption_Time", "Encryption_Size", "Decryption_Time"])
    run_results = []

    a = np.random.randint((n, m)) * scale
    b = np.random.randint((m, n)) * scale
    
    # Size_Data: log size of both matrices
    run_results += [a.nbytes + b.nbytes]

    # encrypting matrices, a and b
    start = timer()
    secretKey = get_secret_key()
    
    a_encrypt = encrypt_AES_GCM(a.tobytes(), secretKey)
    b_encrypt = encrypt_AES_GCM(b.tobytes(), secretKey)
    
    # Encryption_Time: log encryption time
    stop = timer()
    run_results += [stop - start]

    # Encryption_Size: log size of encrypted matrices
    run_results += [get_encrypted_size_mat(a_encrypt) + get_encrypted_size_mat(b_encrypt)]

    # decrypting encrypted matrices
    start = timer()
    a_decrypt = np.frombuffer(decrypt_AES_GCM(a_encrypt, secretKey), dtype=int)
    a_decrypt.resize((a.shape)) 
    b_decrypt = np.frombuffer(decrypt_AES_GCM(b_encrypt, secretKey), dtype=int)
    b_decrypt.resize((b.shape))

    res = a_decrypt @ b_decrypt

    # Decryption_Time: log decryption time
    stop = timer()
    run_results += [stop - start]

    # append results of the current run to the results dataframe
    results_dataframe.loc[len(results_dataframe.index)] = run_results

    return results_dataframe


'''
    FHE Matrix Multiplication - Integer
'''


def run_mat_muli_fhe(results_dataframe: pd.DataFrame, n: int, m: int, scale: int, params=None) -> pd.DataFrame:
    # check if the results dataframe has been initialized yet
    if results_dataframe.empty:
        results_dataframe = pd.DataFrame(
            columns=["Encryption_Time_Data", "Encryption_Size_Data", "Processing_Time", "Encryption_Size_Results",
                     "Decryption_Results_Time", "Accuracy"])
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
        results_dataframe = pd.DataFrame(
            columns=["Encryption_Time_Data", "Encryption_Size_Data", "Processing_Time", "Encryption_Size_Results",
                     "Decryption_Results_Time", "Accuracy"])
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
