# Prakash Bhasker
# prakash.p.bhasker@intel.com
# 05/02/2016
# Targets Python 2.7 with pycryptodome

from __future__ import print_function

import argparse
import datetime
import glob
import multiprocessing
import os
import shutil
import struct
import sys
import time
import zipfile

from Crypto.PublicKey import RSA
from Crypto.Cipher import AES, PKCS1_OAEP
from multiprocessing import Pool, Queue, Lock, Manager

# Constants

EXTENSIONS = {'ENCRYPTED'   : '.encrypted',
              'ZIPPED'      : '.zip',
              'CSV'         : '.csv',
              'ARCHIVE'     : '.archive',
              'BINARY'      : '.bin'}

BINARY_DATA_WIDTHS = {'BatchedAccelerometer' : 3,
                      'BatchedGravity'       : 3,
                      'BatchedGyroscope'     : 3,
                      'BatchedMagneticField' : 3,
                      'BatchedLight'         : 1,
                      'BatchedProximity'     : 1,
                      'BatchedStepCount'     : 1}

LOG_LEVELS = {'verb' : 3,
              'info' : 2,
              'warn' : 1,
              'erro' : 0}

AES_CCM_NONCE_LENGTH = 11
AES_KEY_SIZE = 16
CCM_MAC_SIZE = 16
RSA_MESSAGE_SIZE = 256

SUCCESS = 0
FAILED = -1

# Globals
log_level = LOG_LEVELS['verb']

def log(message):
    print('{} {} {}'.format(str(datetime.datetime.now()), os.getpid(), message), end='')

def verb(message):
    if (log_level >= LOG_LEVELS['verb']):
        log('VERB: {}\r\n'.format(message));

def info(message):
    if (log_level >= LOG_LEVELS['info']):
        log('INFO: {}\r\n'.format(message));

def warn(message):
    if (log_level >= LOG_LEVELS['warn']):
        log('WARN: {}\r\n'.format(message));

def erro(message):
    if (log_level >= LOG_LEVELS['erro']):
        log('ERRO: {}\r\n'.format(message));

### Decrypts an encrypted appmon file
#   AppMon files are encrypted via AES/CCM with an 11 byte nonce and a 16 byte MAC
#   AES/CCM key is encrypted using a 2048 bit RSA key and prepended to the encrypted file
#   | 256 byte AES/CCM key encrypted with RSA | 11 byte CCM Nonce | Variable length Ciphertext | 16 byte AES/CCM MAC |
def decrypt_app_mon_file(fin, fout, rsa_key):
    try:
        f = open(fin, 'rb')

        # Check file len. Min file len is RSA_MESSAGE_SIZE + AES_CCM_NONCE_LENGTH + CCM_MAC_SIZE = 283
        if os.path.getsize(fin) < (RSA_MESSAGE_SIZE + AES_CCM_NONCE_LENGTH + CCM_MAC_SIZE):
            warn('bad file size')
            return FAILED

        cipher_rsa_decrypt = PKCS1_OAEP.new(rsa_key)

        aes_key = cipher_rsa_decrypt.decrypt(f.read(RSA_MESSAGE_SIZE))
        nonce = f.read(AES_CCM_NONCE_LENGTH)
        # FIXME dangerous if file is large!
        data = f.read()
        f.close()

        ciphertext = data[:-CCM_MAC_SIZE]
        mac = data[-CCM_MAC_SIZE:]

        cipher = AES.new(aes_key, AES.MODE_CCM, nonce)

        # No hdr used...
        plaintext = cipher.decrypt(ciphertext)
        cipher.verify(mac)
        f = open(fout, 'w+')
        f.write(plaintext)
        f.close()
        verb('Successfully decrypted {}'.format(os.path.basename(fin)))
        return SUCCESS
    except BaseException as e:
        warn('Failed to decrypt {}: {}'.format(fin, str(e)))
        return FAILED

def expand_file(fin, fout):
    try:
        f = open(fin, 'rb')
        z = zipfile.ZipFile(f)
        for name in z.namelist():
            outpath = z.extract(name, fout)
        f.close()
        verb('Successfully expanded {}'.format(os.path.basename(fin)))
        return SUCCESS
    except BaseException as e:
        warn('Failed to expand {}: {}'.format(fin, str(e)))
        return FAILED

### Returns the number of floats in row of binary data
def get_value_count(input_file):
    for key in BINARY_DATA_WIDTHS.keys():
        if input_file.find(key) > -1:
            return BINARY_DATA_WIDTHS[key]
    return FAILED

### Converts binary appmon file to csv
def bin_to_csv(fin, fout):
    try:
        # Look up the number of floats in a row of data for this file
        value_count = get_value_count(fin)

        if value_count < 0:
            warn('Could not find row widith for {}'.format(fin))
            return FAILED

        file_len = os.path.getsize(fin)
        data_read = 0

        inf = open(fin, 'r', buffering = 1 * 1024 * 1024) # 1MB
        outf = open(fout, 'w+')

        # Convert data to CSV row by row
        while data_read < file_len:
            timestamp = inf.read(8)
            data_read += len(timestamp)
            outf.write(str(struct.unpack('>q', timestamp)[0]))
            outf.write(',')
            for i in range(value_count):
                value = inf.read(4)
                data_read += len(value)
                outf.write(str(struct.unpack('>f', value)[0]))
                outf.write(',')
            outf.write('\r\n')
        inf.close()
        outf.close()
        verb('Successfully converted {}'.format(os.path.basename(fin)))
        return SUCCESS
    except BaseException as e:
        warn(str(e))
        return FAILED

def move_file(fin, fout):
    os.rename(fin, fout + '/' + os.path.basename(fin))

# Message loop routine for decrypt workers
def worker_decrypt(q, rsa_key_encoded, lock, results):
    rsa_key = RSA.import_key(rsa_key_encoded)
    failure_count = 0
    success_count = 0
    while not q.empty():
        try:
            args = q.get()
            if SUCCESS == decrypt_app_mon_file(args[0], args[1], rsa_key):
                success_count += 1
            else:
                failure_count += 1
        except BaseException as e:
            # Race between q.empty() and q.get()...
            warn(str(e))
            time.sleep(1)
    lock.acquire()
    results['decrypt_success_count'] += success_count
    results['decrypt_fail_count'] += failure_count
    lock.release()

# Message loop routine for unzip workers
def worker_expand(q, lock, results):
    failure_count = 0
    success_count = 0
    while not q.empty():
        try:
            args = q.get()
            if SUCCESS == expand_file(args[0], args[1]):
                success_count += 1
            else:
                failure_count += 1
        except BaseException as e:
            # Race between q.empty() and q.get()...
            warn(str(e))
            time.sleep(1)
    lock.acquire()
    results['expand_success_count'] += success_count
    results['expand_fail_count'] += failure_count
    lock.release()

# Message loop routine for binary to CSV workers
def worker_convert(q, lock, results):
    failure_count = 0
    success_count = 0
    while not q.empty():
        try:
            args = q.get()
            if SUCCESS == bin_to_csv(args[0], args[1]):
                success_count += 1
            else:
                failure_count += 1
        except BaseException as e:
            # Race between q.empty() and q.get()...
            warn(str(e))
            time.sleep(1)
    lock.acquire()
    results['convert_success_count'] += success_count
    results['convert_fail_count'] += failure_count
    lock.release()

if __name__ == '__main__':
    if 2 != sys.version_info.major or 7 != sys.version_info.minor:
        erro("FATAL - Requires python 2.7.x")
        sys.exit(1)

    parser = argparse.ArgumentParser(description='Decrypts files generated by AppMon and converts binary files to CSV')
    parser.add_argument("key_file", type=str, help='relative or absolute path to RSA private key file')
    parser.add_argument('-s', '--src', dest='src_dir', type=str, help='relative or absolute path to direcotry containing AppMon log files', default='.')
    parser.add_argument('-d', '--dest', dest='dest_dir', type=str, help='relative or absolute path to destination directory', default='.')
    parser.add_argument('-t', '--thread_count', dest='thread_count', type=int, help='number of processing threads', default=4)
    parser.add_argument('-l', '--log_level', dest='log_level', choices=LOG_LEVELS.keys(), type=str, default='info')
    args = parser.parse_args()

    # Set log level
    log_level = LOG_LEVELS[args.log_level]

    # Create Manager and Lock for shared variables
    lock = Lock()
    manager = Manager()

    # Check for pycryptodome install
    try:
        AES.MODE_CCM
    except AttributeError:
        erro('AES CCM Mode not found. is pycryptodome installed?')
        sys.exit(1)

    # Check that key file exists
    if not os.path.exists(args.key_file):
        erro('Key file does not exist')
        sys.exit(1)

    # Read RSA private key from key file
    f = open(args.key_file, 'r')
    priv_key_encoded = f.read()
    f.close()

    rsa_private_key = RSA.import_key(priv_key_encoded)

    # Check if src dir exists
    if not os.path.exists(args.src_dir):
        erro('Source directory does not exist!')
        sys.exit(1)

    # Ensure source directory is a directory
    if not os.path.isdir(args.src_dir):
        erro('Source directory is a file!')
        sys.exit(1)

    # Check is dest_dir is a file
    if os.path.isfile(args.dest_dir):
        erro('destination directory is a file')
        sys.exit(1)

    # Create dest_dir
    if not os.path.exists(args.dest_dir):
        os.mkdir(args.dest_dir)

    # Path to temp dir
    tmp_dir = args.dest_dir + os.sep + 'tmp'

    if os.path.exists(tmp_dir):
        tmp_dir_count = 0
        while os.path.exists('{}{}{}'.format(tmp_dir, '_', tmp_dir_count)):
            tmp_dir_count += 1
            verb('{}{}{} already exists; incrementing path counter'.format(tmp_dir, '_', tmp_dir_count))
        tmp_dir = '{}{}{}'.format(tmp_dir, '_', tmp_dir_count)

    # Create tmp dir
    os.mkdir(tmp_dir)

    # Remove .archive extension from archive files
    archive_files = glob.glob(args.src_dir + os.sep + '*AppMon*{}'.format(EXTENSIONS['ARCHIVE']))

    for archive_file in archive_files:
        if os.path.isfile(archive_file):
            shutil.copyfile(archive_file, tmp_dir + os.sep + os.path.basename(archive_file[:-len(EXTENSIONS['ARCHIVE'])]))

    ### Decrypt files
    encrypted_files = (glob.glob(args.src_dir + os.sep + '*AppMon*{}'.format(EXTENSIONS['ENCRYPTED'])) + glob.glob(tmp_dir + os.sep + '*AppMon*{}'.format(EXTENSIONS['ENCRYPTED'])))
    info("Decrypting {} files".format(len(encrypted_files)))

    decrypt_q = multiprocessing.Queue()

    # Add encrypted file list to Queue
    for encrypted_file in encrypted_files:
        job = [encrypted_file,
               '{}{}{}'.format(tmp_dir, os.sep, os.path.basename(encrypted_file[:-len(EXTENSIONS['ENCRYPTED'])]))]
        decrypt_q.put(job)

    results = manager.dict()
    results['decrypt_success_count'] = 0
    results['decrypt_fail_count'] = 0
    results['expand_success_count']  = 0
    results['expand_fail_count']  = 0
    results['convert_success_count'] = 0
    results['convert_fail_count'] = 0

    p = Pool(args.thread_count, worker_decrypt, (decrypt_q, priv_key_encoded, lock, results,))

    while not decrypt_q.empty():
        time.sleep(1)

    ### Expand files
    zipped_files = (glob.glob(args.src_dir + os.sep + '*AppMon*{}'.format(EXTENSIONS['ZIPPED'])) + glob.glob(tmp_dir + os.sep + '*AppMon*{}'.format(EXTENSIONS['ZIPPED'])))
    info("Expanding {} files".format(len(zipped_files)))

    expand_q = multiprocessing.Queue()

    # Add compressed file list to Queue
    for zipped_file in zipped_files:
        job = [zipped_file,
               '{}{}'.format(tmp_dir, os.sep)]
        expand_q.put(job)

    p = Pool(args.thread_count, worker_expand, (expand_q, lock, results,))

    while not expand_q.empty():
        time.sleep(1)

    ### Convert binary files to csv
    binary_files = glob.glob(args.src_dir + os.sep + '*AppMon*{}'.format(EXTENSIONS['BINARY'])) + glob.glob(tmp_dir + os.sep + '*AppMon*{}'.format(EXTENSIONS['BINARY']))
    info("Converting {} files".format(len(binary_files)))

    convert_q = multiprocessing.Queue()

    # Add binary file list to Queue
    for binary_file in binary_files:
        job = [binary_file,
               '{}{}{}{}'.format(tmp_dir, os.sep, os.path.basename(binary_file[:-len(EXTENSIONS['BINARY'])]), EXTENSIONS['CSV'])]
        convert_q.put(job)

    p = Pool(args.thread_count, worker_convert, (convert_q, lock, results,))

    while not convert_q.empty():
        time.sleep(1)

    ### Move final files to dest_dir
    for csv_file in (glob.glob(args.src_dir + os.sep + '*AppMon*{}'.format(EXTENSIONS['CSV'])) + glob.glob(tmp_dir + os.sep + '*AppMon*{}'.format(EXTENSIONS['CSV']))):
        if os.path.isfile(csv_file):
            move_file(csv_file, args.dest_dir)

    ### Clean up
    shutil.rmtree(tmp_dir)

    ### Dump summary
    info('{}/{} files decrypted'.format(results['decrypt_success_count'], results['decrypt_success_count'] + results['decrypt_fail_count']))
    info('{}/{} files expanded'.format(results['expand_success_count'], results['expand_success_count'] + results['expand_fail_count']))
    info('{}/{} files converted'.format(results['convert_success_count'], results['convert_success_count'] + results['convert_fail_count']))
