import numpy as np
from Crypto.Cipher import AES
from Crypto import Random


def get_encrypted_size(encrypted_data) -> int:
    sum = 0
    for e in encrypted_data:
        sum += e.__sizeof__()
    return sum


def get_encrypted_size_elementwise(encrypted_data) -> int:
    sum = 0
    for row in encrypted_data:
        sum += get_encrypted_size(row)
    return sum


def percent_error_matrix(mat_ref, mat_calculated) -> float:
    mat_diff = mat_ref - mat_calculated
    mat_error = (mat_diff / mat_ref) * 100
    return np.average(mat_error)



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