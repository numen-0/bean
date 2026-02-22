from tests.utils import BaseTest
import bean.core as bean

class TestApp(BaseTest):

    def test_initialization_sets_globals(self):
        class App(bean.BeanApp):
            def run(self) -> int:
                return 0

        App(name="test-app", debug=False)

        self.assertTrue(bean.BeanApp._initialized)
        self.assertEqual(bean.BeanApp.NAME, "test-app")
        self.assertFalse(bean.BeanApp.DEBUG)

    def test_double_initialization_raises(self):
        class App(bean.BeanApp):
            def run(self) -> int:
                return 0

        App()
        with self.assertRaises(RuntimeError):
            App()

    def test_access_before_initialization_raises(self):
        # Create dummy subclass but DO NOT instantiate
        class App(bean.BeanApp):
            def run(self) -> int:
                return 0

        # Directly access attribute through class
        with self.assertRaises(RuntimeError):
            _ = App.NAME

        # Directly access attribute through instance bypassing __init__
        instance = object.__new__(App)

        with self.assertRaises(RuntimeError):
            _ = instance.NAME
            
        instance = App()
        try:
            _ = instance.NAME
        except Exception as e:
            self.fail(f"Unexpected exception raised: {e}")

    def test_abstract_run_required(self):
        class BadApp(bean.BeanApp):
            pass

        with self.assertRaises(TypeError):
            BadApp() # type: ignore

    def test_default_values(self):
        class App(bean.BeanApp):
            def run(self) -> int:
                return 0

        App()

        self.assertEqual(bean.BeanApp.NAME, "bean-app")
        self.assertFalse(bean.BeanApp.DEBUG)

