# Copyright (C) 2023-2024 Björn A. Lindqvist <bjourne@gmail.com>
from pathlib import Path

TOOLS = ['compiler_c', 'pkgconfig']
def options(ctx):
    for t in TOOLS:
        ctx.load_tool(t)

def configure(ctx):
    for t in TOOLS:
        ctx.load_tool(t)
    ctx.check(lib = 'm', mandatory = False)
    ctx.find_program('aoc', var='AOC', mandatory = False)
    ctx.find_program('aocl', var='AOCL', mandatory = False)
    if ctx.env['AOCL']:
        ctx.check_cfg(path = 'aocl', args = ['compile-config'],
                      package = '', uselib_store = 'AOCL', mandatory = False)
        ctx.check_cfg(path = 'aocl', args = ['linkflags'],
                      package = '', uselib_store = 'AOCL', mandatory = False)
    ctx.check(lib = 'OpenCL', mandatory = True, use = ['AOCL'])
    ctx.define('_GNU_SOURCE', 1)

    c_flags = [
        '-Wall', '-Werror', '-Wextra',
        '-O3',
        #'-ffp-contract=off',
        '-fomit-frame-pointer',
        '-march=native', '-mtune=native'
    ]
    ctx.env.append_unique('CFLAGS', c_flags)

def build(ctx):
    source = [str(p) for p in Path('src').glob('*.c')]
    ctx(features = ["c", "cprogram"],
        source = source,
        target = 'csim',
        install_path = None,
        use = ['AOCL', 'M', 'OPENCL'])
