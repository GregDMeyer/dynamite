
import unittest as ut
from test_multiply import *

#init_args = ['-start_in_debugger', 'noxterm']
init_args = []

from dynamite import config
config.initialize(init_args)
config.shell = 'cpu'
config.L = 10
ut.main()
