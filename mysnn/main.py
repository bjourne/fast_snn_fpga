# Copyright (C) 2023-2024 Björn A. Lindqvist <bjourne@gmail.com>
from mysnn import builders
from sys import argv

COMMANDS = {
    'build' : builders.build
}

def main():
    n_args = len(argv)
    cmd = argv[1]
    fun = COMMANDS[cmd]
    fun(argv[2:])

main()
