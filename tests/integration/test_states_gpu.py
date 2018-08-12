
from dynamite import config
config.initialize(['-vec_type','cuda','-vec_view','::ascii_info'])

from test_states import ToNumpy
import unittest as ut

ut.main()
