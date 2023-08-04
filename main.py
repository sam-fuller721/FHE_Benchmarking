import numpy as np
from Pyfhel import Pyfhel, PyCtxt
from sklearn import datasets
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


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


def test_func():
    data, target = datasets.load_iris(return_X_y=True)
    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=.5, random_state=1)
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
        encrypted_prediction = [
            HE.scalar_prod_plain(data, encoded_coef, in_new_ctxt=True).to_bytes()
            for encoded_coef in encoded_coefs
        ]
        predictions.append(encrypted_prediction)
    c1_preds = []
    for prediction in predictions:
        logits = [PyCtxt(bytestring=p, scheme="CKKS", pyfhel=HE) for p in prediction]
        cl = np.argmax([HE.decryptFrac(logit)[0] for logit in logits])
        c1_preds.append(cl)
    print("Accuracy:", metrics.accuracy_score(target_test, y_pred))

def run_linear_reg_test():
    # load and preprocess iris dataset
    iris = datasets.load_iris()
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    target_df = pd.DataFrame(data=iris.target, columns=["species"])
    iris_df = (iris_df - iris_df.mean(0)) / (iris_df.std(0))

    data, target = datasets.load_iris(return_X_y=True)
    acc = 0
    runs = 100
    test_perc = .5
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
    for seed in range(0, runs):
        data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=test_perc, random_state=seed)
        # encrypt test data
        iris_np = np.append(data_test, np.ones((len(data_test), 1)), -1)  # append 1 for bias term
        encrypted_iris = [HE.encryptFrac(row).to_bytes() for row in iris_np]
        # create linear regressor
        lin_reg = LogisticRegression()
        # run encrypted data through linear regression model
        y_pred = lin_reg.fit(data_train, target_train).predict(encrypted_iris)
        # decrypt data
        for pred in y_pred:
            logits = [PyCtxt(bytestring=p, scheme="CKKS", pyfhel=HE) for p in pred]
            cl = np.argmax([HE.decryptFrac(logit)[0] for logit in logits])
            print(cl)

        # calculate accuracy
        #acc += metrics.accuracy_score(target_test, y_pred)
    # save iris dataset
    iris_df = pd.concat([iris_df, target_df], axis=1)


if __name__ == "__main__":
    test_func()
