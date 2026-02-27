from typing import NoReturn, override
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

    def assertUnreachable(
        self,
        msg: str = "This code path should be unreachable"
    ) -> NoReturn:
        """Mark a code path as unreachable. Fails if executed."""
        self.fail(msg)

    def todo(self, msg: str = "TODO: implement this test"):
        """Mark a test as a TODO."""
        self.fail(msg)
