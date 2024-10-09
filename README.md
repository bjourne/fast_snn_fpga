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

* Python 3 with the following packages:
** humanize
** rich
* OpenCL

## Building

Look through mysnn/pod2014.py and configure relevant network
parameters. Then generate network data:

    PYTHONPATH=. python mysnn/main.py build numpy

Grab some coffee because it takes forever. Then build the simulator:

    ./waf configure build
