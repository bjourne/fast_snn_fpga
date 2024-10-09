# Fast Spiking Neural Network Simulation Algorithms on FPGAs

Code for the paper "Algorithms for Fast Spiking Neural Network
Simulation on FPGAs".

If you find this work useful, consider citing us:

```
@article{
    bal_ap_2024,
    TODO
}
```

## Dependencies

* Python 3
* The Python packages[humanize](https://pypi.org/project/humanize/),
  [rich](https://github.com/Textualize/rich), and
  [numpy](https://numpy.org/)
* OpenCL

## Building and Running

Look through mysnn/pod2014.py and configure relevant network
parameters. Then generate network data:

    PYTHONPATH=. python mysnn/main.py build numpy

Grab some coffee because it takes forever. Then build the simulator:

    ./waf configure build

You can simulate on the CPU:

    ./build/csim networks/numpy 1000 cpu s

You can simulate using OpenCL if you have a suitable OpenCL runtime
installed:

    ./build/csim networks/numpy 1000 opencl cpu/s 0 0 kernels/cpu.cl

To run on an FPGA use something like:

    /build/csim networks/1.00 10000 opencl fpga/horiz/multi/d 1 0 path/to/file.aocx

Where `path/to/file.aocx` is an FPGA image built with the aoc
compiler. A typical aoc build command:

    aoc -bsp-flow=flat -seed=xxxx -parallel=16 \
        -ffp-contract=fast -ffp-reassociate -O3 -I../.. \
        -D RECORD_SPIKES=0 -D WIDTH=16 -D SYN_UNROLL=2 -D SYN_ALIGN=16 \
        -D N_LANES=16 -D N_FRONT=16 -D USE_DOUBLES=1 -v \
        -board=B2E2_8GBx4 jit_mono.cl
