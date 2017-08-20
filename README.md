ARC Vision Process
===============================

version number: 0.0.1
author: Andrew White

Overview
--------

Vision process

Installation / Usage
--------------------
Requires opencv. Also requires (currently) development version of zmq. To install this, run

    pip install git+git://github.com/zeromq/pyzmq.git@v17.0.0b1

To install package in edit mode use pip:

    $ pip install -e .


Execute
----------------

Run from `ipython` or `python` to get detailed error messages

    import arcvision
    arcvision.main()