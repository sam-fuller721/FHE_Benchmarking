import pandas as pd
import argparse 
import json
import os
from tqdm import tqdm
from LogisticRegression import *
from MatrixMult import *
from MatrixScale import *
from BasicOperators import *


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
        "mat_scale_float_AES": run_mat_scalef_AES,
        "mat_scale_int_AES": run_mat_scalei_AES,
        "mat_scale_float_FHE": run_mat_scalef_FHE,
        "mat_scale_int_FHE": run_mat_scalei_FHE,
        "scalar_addf": run_scalar_addf,
        "scalar_addi": run_scalar_addi,
        "scalar_subf": run_scalar_subf,
        "scalar_subi": run_scalar_subi,
        "scalar_multf": run_scalar_multf,
        "scalar_multi": run_scalar_multi,
        "scalar_divf": run_scalar_divf,
        "scalar_divi": run_scalar_divi,
        "scalar_addf_FHE": run_scalar_addf_FHE,
        "scalar_addi_FHE": run_scalar_addi_FHE,
        "scalar_subf_FHE": run_scalar_subf_FHE,
        "scalar_subi_FHE": run_scalar_subi_FHE,
        "scalar_multf_FHE": run_scalar_multf_FHE,
        "scalar_multi_FHE": run_scalar_multi_FHE,
        "scalar_divf_FHE": run_scalar_divf_FHE,
        "scalar_divi_FHE": run_scalar_divi_FHE,
        "repeated_multf_FHE_relin": run_repeated_multf_FHE_relin,
        "repeated_multf_FHE_norelin": run_repeated_multf_FHE_norelin
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
                if "repetitions" in test:
                    args += [test["repetitions"]]
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
                args[0].to_csv(f'{test_file_path}/{test["type"]}_results_run{cnt}.csv')
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
