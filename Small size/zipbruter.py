#!/usr/bin/python3

"""
Password cracker for ZIP files. Uses brute
force attack vector for this, so you must have
strong wordlist.

Usage:
    python3 zipbruter.py -f <encrypted_zip_file> -w <wordlist> -t <threads>
"""

from sys import exit as exit_
from os.path import isfile
from argparse import ArgumentParser
from _thread import start_new_thread
from queue import Queue
from zipfile import is_zipfile, ZipFile, BadZipfile


class ZipBruter:

    def __init__(self, file, word_list, threads) -> None:
        self.file = file
        self.word_list = word_list
        self.threads = threads

        self.queue = Queue()

    def worker(self) -> None:
        while True:
            passwd = self.queue.get()
            self.queue.task_done()

            if passwd is None:
                break

            try:
                with ZipFile(self.file) as zipfile:
                    zipfile.extractall(pwd=passwd.encode())
                print('Found passwd: %s' % passwd)
            except (RuntimeError, BadZipfile):
                pass

    def start_workers(self) -> None:
        for _ in range(self.threads):
            start_new_thread(self.worker, ())

    def main(self) -> None:
        self.start_workers()

        for target_passwd in self.read_wordlist():
            self.queue.put(target_passwd)

        for _ in range(self.threads):
            self.queue.put(None)

        self.queue.join()

    def read_wordlist(self) -> str:
        with open(self.word_list, 'r') as file:
            for line in file.readlines():
                yield line.strip()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-f', '--file', type=str, help='Target encrypted zip file.')
    parser.add_argument('-w', '--word-list', type=str, help='Wordlist to be used.')
    parser.add_argument('-t', '--threads', type=int, default=4, help='Thread count.')
    args = parser.parse_args()

    if not args.file or not args.word_list:
        exit_(1)

    if not is_zipfile(args.file):
        exit_(1)

    if not isfile(args.word_list):
        exit_(1)

    bruter = ZipBruter(args.file, args.word_list, args.threads)
    bruter.main()
