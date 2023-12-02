
import unittest as ut
from dynamite import tools
import dynamite_test_runner as dtr

class Tools(dtr.DynamiteTestCase):
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
        self.assertTrue(isinstance(tools.get_memory_usage(group_by='all'), float))
        self.assertTrue(isinstance(tools.get_memory_usage(group_by='rank'), float))
        self.assertTrue(isinstance(tools.get_memory_usage(group_by='node'), float))

    def test_max_memory(self):
        tools.track_memory()
        self.assertTrue(isinstance(tools.get_memory_usage(group_by='all', max_usage=True), float))
        self.assertTrue(isinstance(tools.get_memory_usage(group_by='rank', max_usage=True), float))
        self.assertTrue(isinstance(tools.get_memory_usage(group_by='node', max_usage=True), float))


if __name__ == '__main__':
    dtr.main()
