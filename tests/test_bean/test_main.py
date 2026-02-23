from tests.utils import BaseTest
import bean.core as bean

class TestMain(BaseTest):

    def test_basic_run_success(self):
        # Mock BeanApp subclass
        class MockApp(bean.BeanApp):
            def startup(self):
                return True
            def run(self):
                return 42
            def shutdown(self):
                return True

        app = MockApp()

        with self.assertRaises(SystemExit) as cm:
            bean.main(app)

        self.assertEqual(cm.exception.code, 42)

    def test_startup_failure(self):
        class MockApp(bean.BeanApp):
            def startup(self):
                return False
            def run(self):
                return 0
            def shutdown(self):
                return True

        app = MockApp()

        with self.assertRaises(SystemExit) as cm:
            bean.main(app)

        self.assertEqual(cm.exception.code, 1)

    def test_shutdown_failure(self):
        class MockApp(bean.BeanApp):
            def startup(self):
                return True
            def run(self):
                return 7
            def shutdown(self):
                return False

        app = MockApp()

        with self.assertRaises(SystemExit) as cm:
            bean.main(app)

        self.assertEqual(cm.exception.code, 1)

