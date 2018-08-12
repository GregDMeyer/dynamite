
import unittest as ut
from test_multiply import *

init_args = [
    '-vec_type','cuda',
    '-mat_type','aijcusparse',
#    '-mat_view','-vec_view'
]

from dynamite import config
config.initialize(init_args)
config.L = 10
ut.main()
