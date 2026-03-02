from tests.utils import BaseTest
import bean.core as bean

class TestApp(BaseTest):

    def test_initialization(self):
        class App(bean.BeanApp):
            def run(self) -> int:
                return 0

        name = "test-app"
        app = App(name=name)

        self.assertEqual(app.name, name)

    def test_abstract_run_required(self):
        class BadApp(bean.BeanApp):
            pass

        with self.assertRaises(TypeError):
            BadApp() # type: ignore

    def test_default_values(self):
        class App(bean.BeanApp):
            def run(self) -> int:
                return 0

        app = App()

        self.assertEqual(app.name, "bean-app")

