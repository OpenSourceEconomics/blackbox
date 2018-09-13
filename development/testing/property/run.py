#!/usr/bin/env python
"""This script allows to create and run the regression test."""
from blackbox.tests.test_blackbox import *

for i in range(1000):

    print(' ... iteration ', str(i), '\n\n')

    test_1()

    test_3()

    test_4()

    test_6()

    test_7()
