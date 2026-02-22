from tests.utils import BaseTest
from bean.core import Pipe, Success

class TestPipes(BaseTest):

    def test_basic_pipe(self):
        pipe = Pipe().map(lambda x: x + 1)
        self.assertCases([
            (pipe(1), 2),
            (pipe(10), 11),
            (pipe(10), 11),
        ], lambda r: r.value)

    def test_pipe_composition(self):
        pipe = Pipe().map(lambda x: x + 1) | Pipe().map(lambda x: (x * 2))
        self.assertEqual(pipe(1), (4, True))

    def test_short_circuit(self):
        pipe = Pipe() | (lambda x: (x * 2, False)) | Pipe().map(lambda x: x + 1)
        self.assertEqual(pipe(1), (2, False))

    def test_map_and_guard(self):
        cases = (
            (1, (2, False)),
            (4, (8, True)),
            (-3, (-6, False)),
        )
        pipe = (
            Pipe[int, int]()
                .map(lambda x: x * 2)
                .guard(lambda x: x > 5)
        )
        self.assertCases(cases, pipe)

        cases = (
            (1, (15, False)),
            (4, (30, True)),
            (-3, (5, False)),
        )
        pipe = (
            Pipe()
                .fallback(lambda x: Success(x + 2, x > 0), 1)
                .map(lambda x: x * 5)
                .guard(lambda x: x > 20)
        )
        self.assertCases(cases, pipe)

    def test_peek(self):
        seen = []

        pipe = (
            Pipe()
                .map(lambda x: x + 1)
                .peek(lambda x: seen.append(x))
                .map(lambda x: x * 2)
        )

        res = pipe(3)

        self.assertEqual(res, (8, True))
        self.assertEqual(seen, [4])

    def test_fallback(self):
        pipe = Pipe().fallback(lambda x: (x * 2, x > 0), fb=99)

        self.assertEqual(pipe(2), (4, True))    # success path
        self.assertEqual(pipe(-2), (99, True))  # fallback path

    def test_branch(self):
        pipe = (
            Pipe()
                .branch(
                    cond_fn=lambda x: x > 0,
                    success_fn=lambda x: (x * 10, True),
                    fail_fn=lambda _: (-1, False),
                )
        )

        self.assertEqual(pipe(3), (30, True))
        self.assertEqual(pipe(-3), (-1, False))

    def test_branch_passthrough(self):
        pipe = Pipe().branch(lambda x: x > 0)

        self.assertEqual(pipe(5), (5, True))
        self.assertEqual(pipe(-5), (-5, False))

    def test_retry_success(self):
        attempts = {"count": 0}

        def flaky(x):
            attempts["count"] += 1
            if attempts["count"] < 3:
                return (x, False)
            return (x * 2, True)

        pipe = Pipe().retry(flaky, attempts=3, delay=0)

        self.assertEqual(pipe(5), (10, True))
        self.assertEqual(attempts["count"], 3)

    def test_retry_fail(self):
        def always_fail(x):
            return (x, False)

        pipe = Pipe().retry(always_fail, attempts=3, delay=0)

        self.assertEqual(pipe(5), (5, False))

    def test_trigger_pass_through(self):
        pipe = (
            Pipe()
                .map(lambda x: x * 2)
                .trigger(lambda x: x > 10)   # trigger only if >10
        )

        self.assertCases([
            (pipe(2), (4, True)),   # 4  <= 10 -> ok
            (pipe(5), (10, True)),  # 10 <= 10 -> ok
        ], lambda r: r)

    def test_trigger_raises_default(self):
        pipe = (
            Pipe()
                .map(lambda x: x * 2)
                .trigger(lambda x: x > 5)
        )

        with self.assertRaises(Exception):
            pipe(4)  # 8 > 5 -> should raise

    def test_trigger_custom_exception(self):
        pipe = (
            Pipe()
                .trigger(lambda x: x < 0, ValueError)
        )

        with self.assertRaises(ValueError):
            pipe(-1)

    def test_trigger_custom_message(self):
        pipe = (
            Pipe()
                .trigger(lambda x: x == 42, RuntimeError, "no 42 allowed")
        )

        with self.assertRaises(RuntimeError) as ctx:
            pipe(42)

        self.assertIn("no 42 allowed", str(ctx.exception))

    def test_trigger_short_circuits_before_next_stage(self):
        pipe = (
            Pipe()
                .trigger(lambda x: x > 0)
                .map(lambda x: x * 999)  # should never run
        )

        with self.assertRaises(Exception):
            pipe(1)

