import numpy as np
from Pyfhel import Pyfhel, PyCtxt
from sklearn import datasets
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import timeit


def test_fhe():
    print("1. Import Pyfhel class, and numpy for the inputs to encrypt.")
    HE = Pyfhel()           # Creating empty Pyfhel object
    HE.contextGen(scheme='bfv', n=2**14, t_bits=20)  # Generate context for 'bfv'/'ckks' scheme
                            # The n defines the number of plaintext slots.
                            #  There are many configurable parameters on this step
                            #  More info in Demo_2, Demo_3, and Pyfhel.contextGen()
    HE.keyGen()
    print("2. Context and key setup")
    print(HE)

    integer1 = np.array([127], dtype=np.int64)
    integer2 = np.array([-2], dtype=np.int64)
    ctxt1 = HE.encryptInt(integer1) # Encryption makes use of the public key
    ctxt2 = HE.encryptInt(integer2) # For integers, encryptInt function is used.
    print("3. Integer Encryption, ")
    print("    int ",integer1,'-> ctxt1 ', type(ctxt1))
    print("    int ",integer2,'-> ctxt2 ', type(ctxt2))

    print(ctxt1)
    print(ctxt2)

    ctxtSum = ctxt1 + ctxt2         # `ctxt1 += ctxt2` for inplace operation
    ctxtSub = ctxt1 - ctxt2         # `ctxt1 -= ctxt2` for inplace operation
    ctxtMul = ctxt1 * ctxt2         # `ctxt1 *= ctxt2` for inplace operation
    print("4. Operating with encrypted integers")
    print(f"Sum: {ctxtSum}")
    print(f"Sub: {ctxtSub}")
    print(f"Mult:{ctxtMul}")

    resSum = HE.decryptInt(ctxtSum) # Decryption must use the corresponding function
                                    #  decryptInt.
    resSub = HE.decryptInt(ctxtSub)
    resMul = HE.decryptInt(ctxtMul)
    print("#. Decrypting result:")
    print("     addition:       decrypt(ctxt1 + ctxt2) =  ", resSum)
    print("     substraction:   decrypt(ctxt1 - ctxt2) =  ", resSub)
    print("     multiplication: decrypt(ctxt1 + ctxt2) =  ", resMul)


def run_mat_mul(n: int, m: int, scale: int):
    a = np.random.random((n, m)) * scale
    b = np.random.random((m, n)) * scale
    res = a @ b


def run_mat_mul_fhe(n: int, m: int, scale: int):
    HE = Pyfhel()
    ckks_params = {
        "scheme": "CKKS",
        "n": 2 ** 14,
        "scale": 2 ** 30,
        "qi_sizes": [60, 30, 30, 30, 60]
    }
    HE.contextGen(**ckks_params)
    HE.keyGen()
    HE.relinKeyGen()
    HE.rotateKeyGen()


    a_mat = np.random.random((n, m)) * scale
    b_mat = (np.random.random((m, n)) * scale)
    print(a_mat @ b_mat)

    a_enc = [HE.encryptFrac(np.array(row)) for row in a_mat]
    b_enc = [HE.encryptFrac(np.array(col)) for col in b_mat.T]
    res = []
    for a_row in a_enc:
        sub_res = []
        for b_col in b_enc:
            sub_res.append(HE.scalar_prod(a_row, b_col, in_new_ctxt=True))
        res.append(sub_res)

    for rows in res:
        for coef in rows:
            print(HE.decryptFrac(coef)[0])


def run_logistic_reg_fhe():
    data, target = datasets.load_iris(return_X_y=True)
    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=.5, random_state=None)
    lin_reg = LogisticRegression()
    y_pred = lin_reg.fit(data_train, target_train).predict(data_test)
    # encrypt test data
    HE = Pyfhel()
    ckks_params = {
        "scheme": "CKKS",
        "n": 2 ** 14,
        "scale": 2 ** 30,
        "qi_sizes": [60, 30, 30, 30, 60]
    }
    HE.contextGen(**ckks_params)
    HE.keyGen()
    HE.relinKeyGen()
    HE.rotateKeyGen()
    data_appended = np.append(data_test, np.ones((len(data_test), 1)), -1)
    encrypted_test_data = [HE.encryptFrac(row) for row in data_appended]
    # encrypt coefficients from trained model
    coefs = []
    for i in range(0, 3):
        coefs.append(np.append(lin_reg.coef_[i], lin_reg.intercept_[i]))
    encoded_coefs = [HE.encodeFrac(coef) for coef in coefs]

    predictions = []
    for data in encrypted_test_data:
        # assumes we know nothing about the data we're inferencing on, including the number of features
        # could use n_elements=data_appended.shape[1] to dramatically improve performance
        encrypted_prediction = [
            HE.scalar_prod_plain(data, encoded_coef, in_new_ctxt=True)
            for encoded_coef in encoded_coefs
        ]
        predictions.append(encrypted_prediction)
    c1_preds = []
    for prediction in predictions:
        cl = np.argmax([HE.decryptFrac(logit)[0] for logit in prediction])
        c1_preds.append(cl)
    print("Accuracy:", metrics.accuracy_score(target_test, y_pred))


if __name__ == "__main__":
    run_mat_mul_fhe(3, 3, 10)
    #print(timeit.timeit("run_logistic_reg_fhe()", setup="from __main__ import run_logistic_reg_fhe", number=10))
