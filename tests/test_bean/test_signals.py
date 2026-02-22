import signal
from unittest.mock import MagicMock
from tests.utils import BaseTest
import bean.core as bean

class TestSignals(BaseTest):

    def test_shutdown_flag_set(self):
        # Directly call the handler to simulate signal
        called = {}
        def cb(signum: int, frame):
            called["sig"] = signum
        
        bean.install_signal_handlers(cb=cb) # type: ignore
        # Grab the handler that was installed for SIGINT
        sigint_handler = signal.getsignal(signal.SIGINT)
        self.assertIsNotNone(sigint_handler)
        sigint_handler(signal.SIGINT, None) # type: ignore

        # The shutdown flag should be set
        self.assertTrue(bean.shutdown_requested())
        self.assertEqual(called.get("sig", None), signal.SIGINT)

    def test_force_exit_warning(self):
        # Can patch Log.warning to check that second Ctrl+C triggers warning
        from bean.core import Log
        log_mock = MagicMock()
        Log.warning = log_mock

        bean.install_signal_handlers()
        sigint_handler = signal.getsignal(signal.SIGINT)
        self.assertIsNotNone(sigint_handler)

        # first Ctrl+C
        sigint_handler(signal.SIGINT, None) # type: ignore
        # second Ctrl+C triggers warning
        sigint_handler(signal.SIGINT, None) # type: ignore

        log_mock.assert_called_with(
            "shutdown already in progress, press Ctrl+C again to force exit"
        )

