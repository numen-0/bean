from tests.utils import BaseTest
from bean.core import Cmd, sh, stdout, stderr, tee, cat

class TestShell(BaseTest):

    # cmd

    def test_cmd_success(self):
        res = Cmd(["echo", "hello"])()
        self.assertTrue(res.ok)
        self.assertEqual(res.code, 0)
        self.assertIn("hello", res.out)

    def test_cmd_failure(self):
        res = Cmd(["false"])()
        self.assertFalse(res.ok)
        self.assertNotEqual(res.code, 0)

    def test_cmd_bool_and_str(self):
        res = Cmd(["echo", "abc"])()
        self.assertTrue(res)
        self.assertEqual(str(res).strip(), "abc")

    # sh + stdout + stderr
    
    def test_sh_stdout(self):
        pipe = sh(["echo", "hello"]) | stdout()
        self.assertEqual(pipe(None), ("hello\n", True))

    def test_sh_stderr(self):
        pipe = sh(["sh", "-c", "echo error 1>&2"]) | stderr()
        result = pipe(None)
        self.assertTrue(result.ok)
        self.assertIn("error", result.value)

    def test_sh_pipeline_input(self):
        pipe = (
            sh(["echo", "hello"])
            | sh(["grep", "hell"])
            | stdout()
        )
        self.assertEqual(pipe(None), ("hello\n", True))

    # tee

    def test_tee(self):
        import tempfile
        import os

        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.close()

        try:
            pipe = sh(["echo", "tee-test"]) | stdout() | tee(tmp.name)
            result = pipe(None)

            self.assertTrue(result.ok)

            with open(tmp.name) as f:
                content = f.read()

            self.assertIn("tee-test", content)

        finally:
            os.unlink(tmp.name)

    # cat

    def test_cat(self):
        import tempfile
        import os

        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(b"abc")
        tmp.close()

        try:
            pipe = cat(tmp.name) | stdout()
            self.assertEqual(pipe(None), ("abc", True))
        finally:
            os.unlink(tmp.name)

    def test_cat_chain(self):
        import tempfile
        import os

        tmp1 = tempfile.NamedTemporaryFile(delete=False)
        tmp2 = tempfile.NamedTemporaryFile(delete=False)

        tmp1.write(b"a")
        tmp2.write(b"b")

        tmp1.close()
        tmp2.close()

        try:
            pipe = cat(tmp1.name, tmp2.name) | stdout()
            self.assertEqual(pipe(None), ("ab", True))
        finally:
            os.unlink(tmp1.name)
            os.unlink(tmp2.name)

