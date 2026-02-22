from textwrap import dedent
from unittest.mock import patch
import tempfile, os
from tests.utils import BaseTest
import bean.core as bean
from bean.core import BeanConfig, ConfigField

class TestConfig(BaseTest):

    def setUp(self):
        super().setUp()

        class AppConfig(BeanConfig):
            PORT = ConfigField(int, default=8080)
            DEBUG = ConfigField(bool, default=False)
            HOST_NAME = ConfigField(str, required=True)

        bean.BeanConfig._instance = None
        self.Config = AppConfig

    # build

    def test_spec_building(self):
        spec = self.Config.spec()

        self.assertIn("PORT", spec)
        self.assertIn("DEBUG", spec)
        self.assertIn("HOST_NAME", spec)

    def test_class_access_before_build_returns_descriptor(self):
        self.assertIsInstance(self.Config.PORT, bean.BeanConfig._ConfigField)

    # load

    def test_source_priority(self):
        cfg = (
            self.Config
            .load()
            .load_dict({"HOST_NAME": "from_dict", "PORT": 1000})
            .load_dict({"PORT": 2000})
            .build()
        )

        self.assertEqual(cfg.PORT, 2000)
        self.assertEqual(cfg.HOST_NAME, "from_dict")

    def test_skip_missing_file(self):
        loader = self.Config.load().load_json("nonexistent.json")
        self.assertEqual(loader.as_dict(), {})

    def test_build_with_required_missing_raises(self):
        with self.assertRaises(ValueError):
            self.Config.load().build()

    def test_defaults_applied(self):
        cfg = (
            self.Config
            .load()
            .load_dict({"HOST_NAME": "localhost"})
            .build()
        )

        self.assertEqual(self.Config.PORT,  8080)
        self.assertEqual(self.Config.DEBUG, False)
        self.assertEqual(self.Config.HOST_NAME,  "localhost")
        self.assertEqual(cfg.PORT,  8080)
        self.assertEqual(cfg.DEBUG, False)
        self.assertEqual(cfg.HOST_NAME,  "localhost")

    def test_type_casting(self):
        cfg = (
            self.Config
            .load()
            .load_dict({
                "HOST_NAME": "localhost",
                "PORT": "9090",
                "DEBUG": "true",
            })
            .build()
        )

        self.assertEqual(cfg.PORT, 9090)
        self.assertTrue(cfg.DEBUG)

    def test_invalid_cast_error(self):
        with self.assertRaises(TypeError):
            ( self.Config
                .load()
                .load_dict({
                    "HOST_NAME": "localhost",
                    "PORT": "not-int",
                })
                .build() )

    def test_unknown_keys_ignored(self):
        loader = (
            self.Config
            .load()
            .load_dict({
                "HOST_NAME": "localhost",
                "UNKNOWN": "value"
            })
        )

        d = loader.as_dict()

        self.assertIn("HOST_NAME", d)
        self.assertNotIn("UNKNOWN", d)


    def test_validator(self):
        class C(bean.BeanConfig):
            PORT = bean.ConfigField(int, validator=lambda x: x > 0)

        with self.assertRaises(ValueError):
            C.load().load_dict({"PORT": -1}).build()

    # misc

    def test_as_dict(self):
        loader = (
            self.Config
            .load()
            .load_dict({"HOST_NAME": "localhost"})
        )

        d = loader.as_dict()
        self.assertIn("HOST_NAME", d)

    # sources

    def test_load_dict(self):
        ( self.Config
            .load()
            .load_dict({
                "HOST_NAME": "localhost",
                "PORT": "9090",
                "DEBUG": "true",
            })
            .build())


        self.assertEqual(self.Config.HOST_NAME, "localhost")
        self.assertEqual(self.Config.PORT, 9090)
        self.assertTrue(self.Config.DEBUG)

    def test_load_args(self):
        ( self.Config
            .load()
            .load_args([
                "--host-name", "localhost",
                "--port", "9090",
                "--debug", "true"
            ]).build())

        self.assertEqual(self.Config.HOST_NAME, "localhost")
        self.assertEqual(self.Config.PORT, 9090)
        self.assertTrue(self.Config.DEBUG)

    def test_load_env(self):
        env = {
            "HOST_NAME": "localhost",
            "PORT": "9090",
            "DEBUG": "true",
        }

        with patch.dict(os.environ, env, clear=True):
            self.Config.load().load_env().build()

            self.assertEqual(self.Config.HOST_NAME, "localhost")
            self.assertEqual(self.Config.PORT, 9090)
            self.assertTrue(self.Config.DEBUG)

    def test_load_env_with_prefix(self):
        env = {
            "APP_HOST_NAME": "localhost",
            "APP_PORT": "9090",
        }

        with patch.dict(os.environ, env, clear=True):
            self.Config.load().load_env(prefix="APP_").build()

            self.assertEqual(self.Config.HOST_NAME, "localhost")
            self.assertEqual(self.Config.PORT, 9090)

    def _dump_load(self, content: str, load_fn, suffix: str =".tmp"):
        with (tempfile.NamedTemporaryFile(mode="w",
                                          delete=False,
                                          suffix=suffix) as tmp):
            tmp.write(content)
            tmp.flush()
            path = tmp.name
            tmp.close()

            load_fn(self.Config.load(), path).build()

        self.assertEqual(self.Config.HOST_NAME, "localhost")
        self.assertEqual(self.Config.PORT, 9090)
        self.assertTrue(self.Config.DEBUG)

        os.remove(path)

    def test_load_py_class(self):
        content = dedent("""
            class Config:
                HOST_NAME = "localhost"
                PORT = 9090
                DEBUG = True
        """).strip()

        self._dump_load(content, BeanConfig._ConfigLoader.load_py, ".py")

    def test_load_py_instance(self):
        content = dedent("""
            class _Config:
                HOST_NAME = "localhost"
                PORT = 9090
                DEBUG = True

            Config = _Config()
        """).strip()

        self._dump_load(content, BeanConfig._ConfigLoader.load_py, ".py")

    def test_load_py_dict(self):
        content = dedent("""
            Config = {
                "HOST_NAME": "localhost",
                "PORT": 9090,
                "DEBUG": True,
            }
        """).strip()

        self._dump_load(content, BeanConfig._ConfigLoader.load_py, ".py")

    def test_load_toml(self):
        content = dedent("""
            [app]
            host_name = "localhost"
            port = 9090
            debug = true
        """).strip()

        self._dump_load(content, BeanConfig._ConfigLoader.load_toml, ".toml")

    def test_load_ini(self):
        content = dedent("""
            [app]
            host.name = localhost
            port = 9090
            debug = true
        """).strip()

        def foo(self, path):
            return BeanConfig._ConfigLoader.load_ini(self, path=path,
                                                     section="app")

        self._dump_load(content, foo, ".ini")

    def test_load_json(self):
        content = dedent("""
            {
                "host-name": "localhost",
                "port": "9090",
                "debug": "true"
            }
        """).strip()

        self._dump_load(content, BeanConfig._ConfigLoader.load_json, ".json")

