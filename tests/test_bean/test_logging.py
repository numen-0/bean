from tests.utils import BaseTest, DummyApp
from bean.core import Log, Logger
import tempfile
import os

class TestLogging(BaseTest):

    def setUp(self):
        super().setUp()

        global Log
        Log = Logger("test-logger")

    def test_logging_capture(self):
        DummyApp(name="test-app", debug=True)

        # Initialize logger with custom name and debug
        logger = Log.child("test")
        logger.set_level(Logger.Level.DEBUG)

        self.assertEqual(logger.name, "test")
        self.assertEqual(logger.level, Logger.Level.DEBUG)

        # Check child logger inherits level and handlers
        child = logger.child("child")
        self.assertTrue(child.name.endswith("child"))
        self.assertEqual(child.level, logger.level)
        self.assertEqual(child.handlers, logger.handlers)

    def test_logger_name_and_level(self):
        DummyApp(name="test-app", debug=True)

        # Initialize logger with custom name and debug
        logger = Log.child("test")
        logger.set_level(Logger.Level.DEBUG)

        self.assertEqual(logger.name, "test")
        self.assertEqual(logger.level, Logger.Level.DEBUG)

        # Check child logger inherits level and handlers
        child = logger.child("child")
        self.assertTrue(child.name.endswith("child"))
        self.assertEqual(child.level, logger.level)
        self.assertEqual(child.handlers, logger.handlers)

    def test_log_file_handler(self):
        # Create a temp file for logs
        tmpfile = tempfile.NamedTemporaryFile(delete=False)
        tmpfile.close()  # close so logging can open it

        DummyApp()

        try:
            logger = Log.child("test-file")

            # Add file handler
            logger.set_handlers([Logger.FileHandler(tmpfile.name, Logger.fmt_basic)])

            # Emit log
            logger.info("File test")

            # Check file was created
            self.assertTrue(os.path.exists(tmpfile.name))

            # Check file contains log text
            with open(tmpfile.name) as f:
                content = f.read()
                self.assertIn("File test", content)

        finally:
            os.unlink(tmpfile.name)
