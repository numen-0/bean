import tempfile, os
from enum import Enum
from textwrap import dedent
from unittest.mock import patch

from tests.utils import BaseTest
import bean.core as bean
from bean.core import BeanConfig, ConfigField

class Stage(Enum):
    DEV = "dev"
    PROD = "prod"
    TEST = "test"

class TestConfig(BaseTest):

    def setUp(self):
        super().setUp()

        class AppConfig(BeanConfig):
            PORT = ConfigField(int, default=8080)
            DEBUG = ConfigField(bool, default=False)
            HOST_NAME = ConfigField(str, required=True)
            STAGE = ConfigField(Stage, default=Stage.DEV)

        bean.BeanConfig._instance = None
        bean.BeanConfig._global_validators = dict()
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
        cfg = ( self.Config
            .load()
            .from_dict({"HOST_NAME": "from_dict", "PORT": 1000})
            .from_dict({"PORT": 2000})
        ).build()

        self.assertEqual(cfg.PORT, 2000)
        self.assertEqual(cfg.HOST_NAME, "from_dict")

    def test_skip_missing_file(self):
        loader = self.Config.load().from_json("nonexistent.json")
        self.assertEqual(loader.as_dict(), {})

    def test_build_with_required_missing_raises(self):
        with self.assertRaises(ExceptionGroup) as cm:
            self.Config.load().build()

        eg = cm.exception
        self.assertTrue(any(isinstance(e, ValueError) for e in eg.exceptions))

    def test_defaults_applied(self):
        cfg = ( self.Config
            .load()
            .from_dict({"HOST_NAME": "localhost"})
        ).build()

        self.assertEqual(cfg.PORT, 8080)
        self.assertEqual(cfg.DEBUG, False)
        self.assertEqual(cfg.HOST_NAME, "localhost")
        self.assertEqual(cfg.STAGE, Stage.DEV)
        self.assertEqual(self.Config.PORT, 8080)
        self.assertEqual(self.Config.DEBUG, False)
        self.assertEqual(self.Config.HOST_NAME, "localhost")
        self.assertEqual(self.Config.STAGE, Stage.DEV)

    def test_type_casting(self):
        cfg = ( self.Config
            .load()
            .from_dict({
                "HOST_NAME": "localhost",
                "PORT": "9090",
                "DEBUG": "true",
                "STAGE": "prod",
            })
        ).build()

        self.assertEqual(cfg.PORT, 9090)
        self.assertTrue(cfg.DEBUG)
        self.assertEqual(cfg.STAGE, Stage.PROD)

    def test_enum_case_insensitive(self):
        cfg = ( self.Config
            .load()
            .from_dict({
                "HOST_NAME": "localhost",
                "PORT": "9090",
                "DEBUG": "false",
                "STAGE": "PrOd",
            })
        ).build()

        self.assertEqual(cfg.STAGE, Stage.PROD)

    def test_invalid_enum_raises(self):
        with self.assertRaises(ExceptionGroup) as cm:
            ( self.Config
                .load()
                .from_dict({
                    "HOST_NAME": "localhost",
                    "PORT": "9090",
                    "DEBUG": "false",
                    "STAGE": "invalid_stage",
                })
            ).build()

        eg = cm.exception
        self.assertTrue(any(isinstance(e, ValueError) for e in eg.exceptions))

    def test_invalid_cast_error(self):
        with self.assertRaises(ExceptionGroup) as cm:
            ( self.Config
                .load()
                .from_dict({
                    "HOST_NAME": "localhost",
                    "PORT": "not-int",
                })
            ).build()

        eg = cm.exception
        self.assertTrue(any(isinstance(e, ValueError) for e in eg.exceptions))

    def test_unknown_keys_ignored(self):
        loader = (
            self.Config
            .load()
            .from_dict({
                "HOST_NAME": "localhost",
                "UNKNOWN": "value"
            })
        )

        d = loader.as_dict()

        self.assertIn("HOST_NAME", d)
        self.assertNotIn("UNKNOWN", d)

    def test_unbound_field_error(self):
        with self.assertRaises(TypeError) as cm:
            class _(bean.BeanConfig):
                NAME = bean.ConfigField(str, required=True)
                PORT = bean.ConfigField(int, required=False)

        err = cm.exception
        self.assertIn("Unbound ConfigField", str(err))
        self.assertIn("_.PORT", str(err))

    def test_short_flag_and_shadow(self):
        class AppConfig(BeanConfig):
            PORT = ConfigField(int, default=8080, short_flag="-p")
            DEBUG = ConfigField(bool, default=False, short_flag="-d")
            HOST_NAME = ConfigField(str, required=True)
            STAGE = ConfigField(Stage, default=Stage.DEV, shadow=["cli"])

        bean.BeanConfig._instance = None
        cfg = AppConfig.load().from_args([
            "-p", "9090",
            "-d",
            "--host-name", "localhost",
            "--stage", "TEST",   # should be shadowed
        ]).build()

        # short flags worked
        self.assertEqual(cfg.PORT, 9090)
        self.assertTrue(cfg.DEBUG)
        self.assertEqual(cfg.HOST_NAME, "localhost")

        # shadowed field should not be applied from CLI
        self.assertEqual(cfg.STAGE, Stage.TEST)  # default retained

    # validator

    def test_validator(self):
        class C(bean.BeanConfig):
            PORT = bean.ConfigField(int, required=True,
                                    validator=lambda x: x > 0)

        with self.assertRaises(ExceptionGroup) as cm:
            C.load().from_dict({"PORT": -1}).build()

        eg = cm.exception
        self.assertTrue(any(isinstance(e, ValueError) for e in eg.exceptions))

    def test_linear_inheritance_fails_base_validator(self):
        """Linear inheritance A -> B, B inherits A's validator, fails validator"""
        class A(BeanConfig):
            VAL = ConfigField(int, required=True)

            @staticmethod
            @BeanConfig.validate("VAL")
            def check_positive(val: int) -> bool:
                return val > 0

        class B(A):
            pass

        with self.assertRaises(ExceptionGroup) as cm:
            B.load().from_dict({"VAL": -1}).build()

        eg = cm.exception
        self.assertTrue(any(isinstance(e, ValueError) for e in eg.exceptions))

    def test_tree_inheritance_mixed(self):
        """Tree inheritance: A -> B & C, B passes, C fails its own validator"""
        class A(BeanConfig):
            VAL = ConfigField(int, required=True)

        @A.validate("VAL")
        def check_positive(val: int) -> bool:
            return val >= 0

        class B(A):
            pass  # inherits validator, should pass

        class C(A):
            pass

        @C.validate("VAL")
        def check_even(val: int) -> bool:
            return val % 2 == 0

        # B should pass
        b_cfg = B.load().from_dict({"VAL": 2}).build()
        self.assertEqual(b_cfg.VAL, 2)

        # C should fail if odd
        with self.assertRaises(ExceptionGroup) as cm:
            C.load().from_dict({"VAL": 3}).build()
        eg = cm.exception
        self.assertTrue(any(isinstance(e, ValueError) for e in eg.exceptions))

    def test_override_validator(self):
        """Override validator in subclass"""
        class A(BeanConfig):
            VAL = ConfigField(int, required=True)

            @staticmethod
            @BeanConfig.validate("VAL")
            def check_positive(val: int) -> bool:
                return val > 0

        @A.validate("VAL")
        def check_positive(val: int) -> bool:
            return True  # override, allow all values

        # A should accept negative
        cfg = A.load().from_dict({"VAL": -100}).build()
        self.assertEqual(cfg.VAL, -100)

    def test_field_normalizer(self):
        class C(bean.BeanConfig):
            RAW = bean.ConfigField(
                str,
                normalizer=lambda s: s.strip().upper(),
                default="  hello  "
            )

        cfg = C.load().from_dict({"RAW": "  world  "}).build()

        self.assertEqual(cfg.RAW, "WORLD")

        cfg_default = C.load().from_dict({}).build()
        self.assertEqual(cfg_default.RAW, "HELLO")

    def test_normalizer_then_validator_failure(self):
        class C(bean.BeanConfig):
            NAME = bean.ConfigField(
                str,
                normalizer=lambda s: s.strip().lower(),
                validator=lambda s: s.isupper(),
            )

        with self.assertRaises(ExceptionGroup) as cm:
            C.load().from_dict({"NAME": "  nope  "}).build()

        eg = cm.exception
        self.assertTrue(any(isinstance(e, ValueError) for e in eg.exceptions))

    # misc

    def test_as_dict(self):
        loader = (
            self.Config
            .load()
            .from_dict({"HOST_NAME": "localhost"})
        )

        d = loader.as_dict()
        self.assertIn("HOST_NAME", d)

    # sources

    def test_from_dict(self):
        ( self.Config
            .load()
            .from_dict({
                "HOST_NAME": "localhost",
                "PORT": "9090",
                "DEBUG": "true",
            })
        ).build()


        self.assertEqual(self.Config.HOST_NAME, "localhost")
        self.assertEqual(self.Config.PORT, 9090)
        self.assertTrue(self.Config.DEBUG)

    def test_from_args(self):
        ( self.Config
            .load()
            .from_args([
                "--host-name", "localhost",
                "--port", "9090",
                "--debug",
                "--stage", "TEST"
            ])
        ).build()

        self.assertEqual(self.Config.HOST_NAME, "localhost")
        self.assertEqual(self.Config.PORT, 9090)
        self.assertTrue(self.Config.DEBUG)
        self.assertEqual(self.Config.STAGE, Stage.TEST)

    def test_from_env(self):
        env = {
            "HOST_NAME": "localhost",
            "PORT": "9090",
            "DEBUG": "true",
            "STAGE": "test",
        }

        with patch.dict(os.environ, env, clear=True):
            self.Config.load().from_env().build()

            self.assertEqual(self.Config.HOST_NAME, "localhost")
            self.assertEqual(self.Config.PORT, 9090)
            self.assertTrue(self.Config.DEBUG)
            self.assertEqual(self.Config.STAGE, Stage.TEST)

    def test_from_env_with_prefix(self):
        env = {
            "APP_HOST_NAME": "localhost",
            "APP_PORT": "9090",
        }

        with patch.dict(os.environ, env, clear=True):
            self.Config.load().from_env(prefix="APP_").build()

            self.assertEqual(self.Config.HOST_NAME, "localhost")
            self.assertEqual(self.Config.PORT, 9090)

    def _dump_load(self, content: str, from_fn, suffix: str =".tmp"):
        with (tempfile.NamedTemporaryFile(mode="w",
                                          delete=False,
                                          suffix=suffix) as tmp):
            tmp.write(content)
            tmp.flush()
            path = tmp.name
            tmp.close()

            from_fn(self.Config.load(), path).build()

        self.assertEqual(self.Config.HOST_NAME, "localhost")
        self.assertEqual(self.Config.PORT, 9090)
        self.assertTrue(self.Config.DEBUG)

        os.remove(path)

    def test_from_py_class(self):
        content = dedent("""
            class Config:
                HOST_NAME = "localhost"
                PORT = 9090
                DEBUG = True
        """).strip()

        self._dump_load(content, BeanConfig._ConfigLoader.from_py, ".py")

    def test_from_py_instance(self):
        content = dedent("""
            class _Config:
                HOST_NAME = "localhost"
                PORT = 9090
                DEBUG = True

            Config = _Config()
        """).strip()

        self._dump_load(content, BeanConfig._ConfigLoader.from_py, ".py")

    def test_from_py_dict(self):
        content = dedent("""
            Config = {
                "HOST_NAME": "localhost",
                "PORT": "9090",
                "DEBUG": True,
                "STAGE": "test",
            }
        """).strip()

        self._dump_load(content, BeanConfig._ConfigLoader.from_py, ".py")

    def test_from_toml(self):
        content = dedent("""
            [app]
            host_name = "localhost"
            port = 9090
            debug = true
            stage = "test"
        """).strip()

        self._dump_load(content, BeanConfig._ConfigLoader.from_toml, ".toml")

    def test_from_ini(self):
        content = dedent("""
            [app]
            host.name = localhost
            port = 9090
            debug = true
            stage = test
        """).strip()

        def foo(self, path):
            return BeanConfig._ConfigLoader.from_ini(self, path=path,
                                                     section="app")

        self._dump_load(content, foo, ".ini")

    def test_from_json(self):
        content = dedent("""
            {
                "host-name": "localhost",
                "port": "9090",
                "debug": "true",
                "stage": "test"
            }
        """).strip()

        self._dump_load(content, BeanConfig._ConfigLoader.from_json, ".json")

