import numpy as np
from Pyfhel import Pyfhel, PyCtxt
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from timeit import default_timer as timer


def run_mat_mulf(n: int, m: int, scale: int) -> float:
    a = np.random.random((n, m)) * scale
    b = np.random.random((m, n)) * scale
    start = timer()
    res = a @ b
    stop = timer()
    return stop - start


def run_mat_muli(n: int, m: int, scale: int) -> float:
    a = np.random.randint(scale, size=(n, m))
    b = np.random.randint(scale, size=(m, n))
    start = timer()
    res = a @ b
    stop = timer()
    return stop - start


def run_mat_muli_bgv(n: int, m: int, scale: int) -> float:
    HE = Pyfhel()
    bgv_params = {
        'scheme': 'BGV',
        'n': 2 ** 13,
        't': 65537,
        't_bits': 20,
        'sec': 128,
    }
    HE.contextGen(**bgv_params)
    HE.keyGen()
    HE.rotateKeyGen()
    HE.relinKeyGen()
    a = np.random.randint(scale, size=(n, m))
    b = np.random.randint(scale, size=(m, n))
    a_enc = [HE.encryptBGV(np.array(row)) for row in a]
    b_enc = [HE.encryptBGV(np.array(col)) for col in b.T]
    start = timer()
    res = []
    for a_row in a_enc:
        sub_res = []
        for b_col in b_enc:
            sub_res.append(HE.scalar_prod(a_row, b_col, in_new_ctxt=True))
        res.append(sub_res)

    for row in res:
        for elem in row:
            print(HE.decryptBGV(elem)[0])
    print(a @ b)
    stop = timer()
    return stop - start


def run_mat_muli_bfv(n: int, m: int, scale: int) -> float:
    HE = Pyfhel()
    bfv_params = {
        'scheme': 'BFV',
        'n': 2 ** 13,
        't': 65537,
        't_bits': 20,
        'sec': 128,
    }
    HE.contextGen(**bfv_params)
    HE.keyGen()
    HE.rotateKeyGen()
    HE.relinKeyGen()
    a = np.random.randint(scale, size=(n, m))
    b = np.random.randint(scale, size=(m, n))
    a_enc = [HE.encryptInt(np.array(row)) for row in a]
    b_enc = [HE.encryptInt(np.array(col)) for col in b.T]
    start = timer()
    res = []
    for a_row in a_enc:
        sub_res = []
        for b_col in b_enc:
            sub_res.append(HE.scalar_prod(a_row, b_col, in_new_ctxt=True))
        res.append(sub_res)

    for row in res:
        for elem in row:
            print(HE.decryptInt(elem)[0])
    print(a @ b)
    stop = timer()

    return stop - start


def run_mat_mulf_ckks(n: int, m: int, scale: int) -> float:
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
    a_enc = [HE.encryptFrac(np.array(row)) for row in a_mat]
    b_enc = [HE.encryptFrac(np.array(col)) for col in b_mat.T]
    start = timer()
    res = []
    for a_row in a_enc:
        sub_res = []
        for b_col in b_enc:
            sub_res.append(HE.scalar_prod(a_row, b_col, in_new_ctxt=True))
        res.append(sub_res)
    stop = timer()
    return stop - start


def run_logistic_reg() -> float:
    data, target = datasets.load_iris(return_X_y=True)
    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=.5, random_state=None)
    # preprocess data
    scaler = preprocessing.StandardScaler().fit(data_train)
    data_train = scaler.transform(data_train)
    data_test = scaler.transform(data_test)
    lin_reg = LogisticRegression()
    # run inference
    start = timer()
    y_pred = lin_reg.fit(data_train, target_train).predict(data_test)
    stop = timer()
    print("Accuracy:", metrics.accuracy_score(target_test, y_pred))
    return stop - start


def run_logistic_reg_fhe() -> float:
    data, target = datasets.load_iris(return_X_y=True)

    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=.5, random_state=None)
    # preprocess data
    scaler = preprocessing.StandardScaler().fit(data_train)
    data_train = scaler.transform(data_train)
    data_test = scaler.transform(data_test)
    lin_reg = LogisticRegression()
    lin_reg.fit(data_train, target_train)
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

    # encrypt coefficients from trained model, start of inferencing process
    start = timer()
    coefs = []
    for i in range(0, 3):
        coefs.append(np.append(lin_reg.coef_[i], lin_reg.intercept_[i]))
    encoded_coefs = [HE.encodeFrac(coef) for coef in coefs]
    # run inference
    predictions = []
    for data in encrypted_test_data:
        encrypted_prediction = [
            HE.scalar_prod_plain(data, encoded_coef, in_new_ctxt=True)
            for encoded_coef in encoded_coefs
        ]
        predictions.append(encrypted_prediction)
    stop = timer()
    # decrypt predictions and check accuracy
    c1_preds = []
    for prediction in predictions:
        cl = np.argmax([HE.decryptFrac(logit)[0] for logit in prediction])
        c1_preds.append(cl)
    print("Accuracy:", metrics.accuracy_score(target_test, c1_preds))
    return stop - start


if __name__ == "__main__":
    runs = 10
    time = 0.0
    for _ in range(0, runs):
        time += run_logistic_reg_fhe()
    print(f'FHE Logistic Regression Average Execution Time ({runs} runs): {time / runs}')

    time = 0.0
    for _ in range(0, runs):
        time += run_logistic_reg()
    print(f'Standard Logistic Regression Average Execution Time ({runs} runs): {time / runs}')

    runs = 100
    time = 0.0
    for _ in range(0, runs):
        time += run_mat_mulf_ckks(5, 5, 10)
    print(f'FHE Matrix Multiplication (5x5) Average Execution Time ({runs} runs): {time / runs}')

    time = 0.0
    for _ in range(0, runs):
        time += run_mat_mulf(5, 5, 10)
    print(f'Standard Matrix Multiplication (5x5) Average Execution Time ({runs} runs): {time / runs}')
