from tests.utils import BaseTest
from bean.core import Success, Result, Predicate

class TestTypes(BaseTest):

    # Success

    def test_success_basic(self):
        cases = [
            (Success.Ok(10), (10, True)),
            (Success.Fail(), (None, False)),
        ]
        self.assertCases(cases, lambda r: r.to_tuple())

    def test_success_bool(self):
        self.assertTrue(Success.Ok(5))
        self.assertFalse(Success.Fail())

    def test_success_eq_tuple(self):
        self.assertEqual(Success.Ok(7), (7, True))
        self.assertEqual(Success.Fail(), (None, False))

    # Result

    def test_result_ok_and_error(self):
        ok = Result.Ok("value")
        err = Result.Error("oops")
        self.assertTrue(ok)
        self.assertFalse(err)
        self.assertEqual(ok.to_tuple(), ("value", True))
        self.assertEqual(err.to_tuple(), ("oops", False))

    def test_result_unwrap(self):
        ok = Result.Ok(42)
        self.assertEqual(ok.unwrap(), 42)
        self.assertEqual(ok.unwrap_or(0), 42)
        err = Result.Error("fail")
        self.assertEqual(err.unwrap_or(100), 100)
        with self.assertRaises(RuntimeError):
            err.unwrap()
        with self.assertRaises(RuntimeError):
            ok.unwrap_err()
        self.assertEqual(err.unwrap_err(), "fail")

    # Predicate

    def test_predicate_basic(self):
        is_even = Predicate(lambda x: x % 2 == 0)
        is_positive = Predicate(lambda x: x > 0)

        self.assertTrue(is_even(4))
        self.assertFalse(is_even(5))
        self.assertTrue(is_positive(10))
        self.assertFalse(is_positive(-1))

    def test_predicate_composition(self):
        is_even = Predicate(lambda x: x % 2 == 0)
        is_positive = Predicate(lambda x: x > 0)
        combined = is_even & is_positive
        self.assertTrue(combined(4))
        self.assertFalse(combined(-4))
        self.assertFalse(combined(3))
        combined_or = is_even | is_positive
        self.assertTrue(combined_or(4))
        self.assertTrue(combined_or(3))
        self.assertFalse(combined_or(-3))
        neg = ~is_even
        self.assertTrue(neg(3))
        self.assertFalse(neg(4))

