from typing import override
import unittest
import bean.core as bean

class DummyApp(bean.BeanApp):
    @override
    def run(self) -> int:
        return 0

class BaseTest(unittest.TestCase):
    """ Base class to support table-driven tests with subTest """

    def setUp(self):
        bean.BeanApp._initialized = True
        bean.BeanApp.NAME = "test-app"
        bean.BeanApp.LOG_FILE = None
        bean.BeanApp.LOG_COLOR = True
        bean.BeanApp._initialized = False

        bean._installed = False
        bean._shutdown_event.clear()

    def assertCases(self, cases, fn):
        """
        cases: iterable of tuples (*args..., expected)
        fn: callable to test
        """
        for case in cases:
            *args, expect = case
            with self.subTest(args=args):
                if isinstance(expect, type) and issubclass(expect, Exception):
                    with self.assertRaises(expect):
                        fn(*args)
                else:
                    self.assertEqual(fn(*args), expect)

