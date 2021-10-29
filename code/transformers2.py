#%%
import collections
import logging
import os
import pathlib
import re
import string
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

import tensorflow_text as text
import tensorflow as tf
# %%
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings
