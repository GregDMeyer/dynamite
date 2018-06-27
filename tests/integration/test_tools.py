
import unittest as ut
from dynamite import tools

class Tools(ut.TestCase):
    '''
    Mostly check that the functions behave as we expect. We can't really verify
    that the output is exactly correct; what we want to ensure is that no
    exceptions are thrown.
    '''

    def test_version(self):
        version = tools.get_version()
        self.assertTrue(isinstance(version, dict))
        for key in ('PETSc', 'SLEPc', 'dynamite'):
            self.assertTrue(key in version)

    def test_version_str(self):
        self.assertTrue(isinstance(tools.get_version_str(), str))

    def test_cur_memory(self):
        self.assertTrue(isinstance(tools.get_cur_memory_usage(), float))

    def test_max_memory(self):
        tools.track_memory()
        self.assertTrue(isinstance(tools.get_max_memory_usage(), float))

if __name__ == '__main__':
    ut.main()
