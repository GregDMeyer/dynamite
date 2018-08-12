
import unittest as ut
from test_multiply import *

init_args = ['-vec_type','cuda']
#init_args += ['-start_in_debugger','noxterm']
#init_args += ['-vec_view']

from dynamite import config
config.initialize(init_args)
config.shell = 'gpu'
config.L = 12
ut.main()
