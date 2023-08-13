import numpy as np
import pandas as pd
from Pyfhel import Pyfhel, PyCtxt
from timeit import default_timer as timer
from HelperFunctions import *

'''
Test for basic operators without encryption
'''


def run_scalar_addf(results_dataframe: pd.DataFrame, scale: int) -> pd.DataFrame:
    # check if the results dataframe has been initialized yet
    if results_dataframe.empty:
        results_dataframe = pd.DataFrame(
            columns=["Size_Data", "Processing_Time", "Size_Results"])
    run_results = []
    a = np.random.random(1) * scale
    b = np.random.random(1) * scale
    run_results += [a.nbytes + b.nbytes]
    start = timer()
    res = a + b
    stop = timer()
    run_results += [stop - start]
    run_results += [res.nbytes]
    results_dataframe.loc[len(results_dataframe.index)] = run_results
    return results_dataframe


def run_scalar_addi(results_dataframe: pd.DataFrame, scale: int) -> pd.DataFrame:
    # check if the results dataframe has been initialized yet
    if results_dataframe.empty:
        results_dataframe = pd.DataFrame(
            columns=["Size_Data", "Processing_Time", "Size_Results"])
    run_results = []
    a = np.array([np.random.randint(1, scale)])
    b = np.array([np.random.randint(1, scale)])
    run_results += [a.nbytes + b.nbytes]
    start = timer()
    res = a + b
    stop = timer()
    run_results += [stop - start]
    run_results += [res.nbytes]
    results_dataframe.loc[len(results_dataframe.index)] = run_results
    return results_dataframe


def run_scalar_subf(results_dataframe: pd.DataFrame, scale: int) -> pd.DataFrame:
    # check if the results dataframe has been initialized yet
    if results_dataframe.empty:
        results_dataframe = pd.DataFrame(
            columns=["Size_Data", "Processing_Time", "Size_Results"])
    run_results = []
    a = np.random.random(1) * scale
    b = np.random.random(1) * scale
    run_results += [a.nbytes + b.nbytes]
    start = timer()
    res = a - b
    stop = timer()
    run_results += [stop - start]
    run_results += [res.nbytes]
    results_dataframe.loc[len(results_dataframe.index)] = run_results
    return results_dataframe


def run_scalar_subi(results_dataframe: pd.DataFrame, scale: int) -> pd.DataFrame:
    # check if the results dataframe has been initialized yet
    if results_dataframe.empty:
        results_dataframe = pd.DataFrame(
            columns=["Size_Data", "Processing_Time", "Size_Results"])
    run_results = []
    a = np.array([np.random.randint(1, scale)])
    b = a
    while b == a:
        b = np.array([np.random.randint(1, scale)])
    run_results += [a.nbytes + b.nbytes]
    start = timer()
    res = a - b
    stop = timer()
    run_results += [stop - start]
    run_results += [res.nbytes]
    results_dataframe.loc[len(results_dataframe.index)] = run_results
    return results_dataframe


def run_scalar_multf(results_dataframe: pd.DataFrame, scale: int) -> pd.DataFrame:
    # check if the results dataframe has been initialized yet
    if results_dataframe.empty:
        results_dataframe = pd.DataFrame(
            columns=["Size_Data", "Processing_Time", "Size_Results"])
    run_results = []
    a = np.random.random(1) * scale
    b = np.random.random(1) * scale
    run_results += [a.nbytes + b.nbytes]
    start = timer()
    res = a * b
    stop = timer()
    run_results += [stop - start]
    run_results += [res.nbytes]
    results_dataframe.loc[len(results_dataframe.index)] = run_results
    return results_dataframe


def run_scalar_multi(results_dataframe: pd.DataFrame, scale: int) -> pd.DataFrame:
    # check if the results dataframe has been initialized yet
    if results_dataframe.empty:
        results_dataframe = pd.DataFrame(
            columns=["Size_Data", "Processing_Time", "Size_Results"])
    run_results = []
    a = np.array([np.random.randint(1, scale)])
    b = np.array([np.random.randint(1, scale)])
    run_results += [a.nbytes + b.nbytes]
    start = timer()
    res = a * b
    stop = timer()
    run_results += [stop - start]
    run_results += [res.nbytes]
    results_dataframe.loc[len(results_dataframe.index)] = run_results
    return results_dataframe


def run_scalar_divf(results_dataframe: pd.DataFrame, scale: int) -> pd.DataFrame:
    # check if the results dataframe has been initialized yet
    if results_dataframe.empty:
        results_dataframe = pd.DataFrame(
            columns=["Size_Data", "Processing_Time", "Size_Results"])
    run_results = []
    a = np.random.random(1) * scale
    b = np.random.random(1) * scale
    run_results += [a.nbytes + b.nbytes]
    start = timer()
    res = a / b
    stop = timer()
    run_results += [stop - start]
    run_results += [res.nbytes]
    results_dataframe.loc[len(results_dataframe.index)] = run_results
    return results_dataframe


def run_scalar_divi(results_dataframe: pd.DataFrame, scale: int) -> pd.DataFrame:
    # check if the results dataframe has been initialized yet
    if results_dataframe.empty:
        results_dataframe = pd.DataFrame(
            columns=["Size_Data", "Processing_Time", "Size_Results"])
    run_results = []
    a = np.array([np.random.randint(1, scale)])
    b = np.array([np.random.randint(1, scale)])
    run_results += [a.nbytes + b.nbytes]
    start = timer()
    res = a / b
    stop = timer()
    run_results += [stop - start]
    run_results += [res.nbytes]
    results_dataframe.loc[len(results_dataframe.index)] = run_results
    return results_dataframe


'''
Test for basic operators with FHE 
'''


def run_scalar_addf_FHE(results_dataframe: pd.DataFrame, scale: int, params=None) -> pd.DataFrame:
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
    a = np.random.random(1) * scale
    b = np.random.random(1) * scale
    start = timer()
    HE.contextGen(**ckks_params)
    HE.keyGen()
    HE.rotateKeyGen()
    HE.relinKeyGen()
    # encrypt vals
    a_enc = HE.encryptFrac(a)
    b_enc = HE.encryptFrac(b)
    stop = timer()
    run_results += [stop - start]
    run_results += [a_enc.__sizeof__() + b_enc.__sizeof__()]
    start = timer()
    res_enc = a_enc + b_enc
    stop = timer()
    run_results += [stop - start]
    run_results += [res_enc.__sizeof__()]
    start = timer()
    res = HE.decryptFrac(res_enc)[0]
    stop = timer()
    run_results += [stop - start]
    run_results += [percent_error_matrix(a + b, res)]
    results_dataframe.loc[len(results_dataframe.index)] = run_results
    return results_dataframe


def run_scalar_addi_FHE(results_dataframe: pd.DataFrame, scale: int, params=None) -> pd.DataFrame:
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
    a = np.array([np.random.randint(1, scale)])
    b = np.array([np.random.randint(1, scale)])
    start = timer()
    HE.contextGen(**bfv_params)
    HE.keyGen()
    HE.rotateKeyGen()
    HE.relinKeyGen()
    # encrypt vals
    a_enc = HE.encryptInt(a)
    b_enc = HE.encryptInt(b)
    stop = timer()
    run_results += [stop - start]
    run_results += [a_enc.__sizeof__() + b_enc.__sizeof__()]
    start = timer()
    res_enc = a_enc + b_enc
    stop = timer()
    run_results += [stop - start]
    run_results += [res_enc.__sizeof__()]
    start = timer()
    res = HE.decryptInt(res_enc)[0]
    stop = timer()
    run_results += [stop - start]
    run_results += [percent_error_matrix(a + b, res)]
    results_dataframe.loc[len(results_dataframe.index)] = run_results
    return results_dataframe


def run_scalar_subf_FHE(results_dataframe: pd.DataFrame, scale: int, params=None) -> pd.DataFrame:
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
    a = np.random.random(1) * scale
    b = np.random.random(1) * scale
    start = timer()
    HE.contextGen(**ckks_params)
    HE.keyGen()
    HE.rotateKeyGen()
    HE.relinKeyGen()
    # encrypt vals
    a_enc = HE.encryptFrac(a)
    b_enc = HE.encryptFrac(b)
    stop = timer()
    run_results += [stop - start]
    run_results += [a_enc.__sizeof__() + b_enc.__sizeof__()]
    start = timer()
    res_enc = a_enc - b_enc
    stop = timer()
    run_results += [stop - start]
    run_results += [res_enc.__sizeof__()]
    start = timer()
    res = HE.decryptFrac(res_enc)[0]
    stop = timer()
    run_results += [stop - start]
    run_results += [percent_error_matrix(a - b, res)]
    results_dataframe.loc[len(results_dataframe.index)] = run_results
    return results_dataframe


def run_scalar_subi_FHE(results_dataframe: pd.DataFrame, scale: int, params=None) -> pd.DataFrame:
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
    a = np.array([np.random.randint(1, scale)])
    b = a
    while b == a:
        b = np.array([np.random.randint(1, scale)])
    start = timer()
    HE.contextGen(**bfv_params)
    HE.keyGen()
    HE.rotateKeyGen()
    HE.relinKeyGen()
    # encrypt vals
    a_enc = HE.encryptInt(a)
    b_enc = HE.encryptInt(b)
    stop = timer()
    run_results += [stop - start]
    run_results += [a_enc.__sizeof__() + b_enc.__sizeof__()]
    start = timer()
    res_enc = a_enc - b_enc
    stop = timer()
    run_results += [stop - start]
    run_results += [res_enc.__sizeof__()]
    start = timer()
    res = HE.decryptInt(res_enc)[0]
    stop = timer()
    run_results += [stop - start]
    run_results += [percent_error_matrix(a - b, res)]
    results_dataframe.loc[len(results_dataframe.index)] = run_results
    return results_dataframe


def run_scalar_multf_FHE(results_dataframe: pd.DataFrame, scale: int, params=None) -> pd.DataFrame:
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
    a = np.random.random(1) * scale
    b = np.random.random(1) * scale
    start = timer()
    HE.contextGen(**ckks_params)
    HE.keyGen()
    HE.rotateKeyGen()
    HE.relinKeyGen()
    # encrypt vals
    a_enc = HE.encryptFrac(a)
    b_enc = HE.encryptFrac(b)
    stop = timer()
    run_results += [stop - start]
    run_results += [a_enc.__sizeof__() + b_enc.__sizeof__()]
    start = timer()
    res_enc = a_enc * b_enc
    stop = timer()
    run_results += [stop - start]
    run_results += [res_enc.__sizeof__()]
    start = timer()
    res = HE.decryptFrac(res_enc)[0]
    stop = timer()
    run_results += [stop - start]
    run_results += [percent_error_matrix(a * b, res)]
    results_dataframe.loc[len(results_dataframe.index)] = run_results
    return results_dataframe


def run_scalar_multi_FHE(results_dataframe: pd.DataFrame, scale: int, params=None) -> pd.DataFrame:
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
    a = np.array([np.random.randint(1, scale)])
    b = np.array([np.random.randint(1, scale)])
    start = timer()
    HE.contextGen(**bfv_params)
    HE.keyGen()
    HE.rotateKeyGen()
    HE.relinKeyGen()
    # encrypt vals
    a_enc = HE.encryptInt(a)
    b_enc = HE.encryptInt(b)
    stop = timer()
    run_results += [stop - start]
    run_results += [a_enc.__sizeof__() + b_enc.__sizeof__()]
    start = timer()
    res_enc = a_enc * b_enc
    stop = timer()
    run_results += [stop - start]
    run_results += [res_enc.__sizeof__()]
    start = timer()
    res = HE.decryptInt(res_enc)[0]
    stop = timer()
    run_results += [stop - start]
    run_results += [percent_error_matrix(a * b, res)]
    results_dataframe.loc[len(results_dataframe.index)] = run_results
    return results_dataframe


def run_scalar_divf_FHE(results_dataframe: pd.DataFrame, scale: int, params=None) -> pd.DataFrame:
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
    a = np.random.random(1) * scale
    b = np.random.random(1) * scale
    start = timer()
    HE.contextGen(**ckks_params)
    HE.keyGen()
    HE.rotateKeyGen()
    HE.relinKeyGen()
    # encrypt vals
    a_enc = HE.encryptFrac(a)
    b_enc = HE.encodeFrac(b)
    stop = timer()
    run_results += [stop - start]
    run_results += [a_enc.__sizeof__() + b_enc.__sizeof__()]
    start = timer()
    res_enc = a_enc / b_enc
    stop = timer()
    run_results += [stop - start]
    run_results += [res_enc.__sizeof__()]
    start = timer()
    res = HE.decryptFrac(res_enc)[0]
    stop = timer()
    run_results += [stop - start]
    run_results += [percent_error_matrix(a / b, res)]
    results_dataframe.loc[len(results_dataframe.index)] = run_results
    return results_dataframe


def run_scalar_divi_FHE(results_dataframe: pd.DataFrame, scale: int, params=None) -> pd.DataFrame:
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
    a = np.array([np.random.randint(1, scale)])
    b = np.array([np.random.randint(1, scale)])
    start = timer()
    HE.contextGen(**bfv_params)
    HE.keyGen()
    HE.rotateKeyGen()
    HE.relinKeyGen()
    # encrypt vals
    a_enc = HE.encryptInt(a)
    b_enc = HE.encodeInt(b)
    stop = timer()
    run_results += [stop - start]
    run_results += [a_enc.__sizeof__() + b_enc.__sizeof__()]
    start = timer()
    res_enc = a_enc / b_enc
    stop = timer()
    run_results += [stop - start]
    run_results += [res_enc.__sizeof__()]
    start = timer()
    res = HE.decryptInt(res_enc)[0]
    stop = timer()
    run_results += [stop - start]
    run_results += [percent_error_matrix(a / b, res)]

    results_dataframe.loc[len(results_dataframe.index)] = run_results
    return results_dataframe
