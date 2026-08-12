import os
from enum import Enum
from unittest.mock import patch

from tests.utils import BaseTest
from bean import config

class Mode(Enum):
    DEV = "dev"
    PROD = "prod"

class Config:
    HOST: str
    PORT: int = 8080
    DEBUG: bool = False
    MODE: Mode = Mode.DEV
    NUMBERS: list[int] = []

    _PRIVATE: str = "this is private"

class TestConfig(BaseTest):

    def test_defaults(self):
        cfg = config.load(
            Config,
            argv=["--host", "localhost"],
            overrides={"HOST": "localhost"},
        )

        self.assertEqual(cfg.HOST, "localhost")
        self.assertEqual(cfg.PORT, 8080)
        self.assertFalse(cfg.DEBUG)
        self.assertEqual(cfg.MODE, Mode.DEV)
        self.assertEqual(cfg.NUMBERS, [])

    def test_missing_required(self):
        with self.assertRaises(ExceptionGroup):
            _ = config.load(Config, argv=[])

    def test_override_priority(self):
        cfg = config.load(
            Config,
            argv=["--port", "7000"],
            overrides={
                "HOST": "localhost",
                "PORT": 9000,
            },
        )

        self.assertEqual(cfg.PORT, 9000)

    @patch.dict(os.environ, {
        "HOST": "env-host",
        "PORT": "9001",
    })
    def test_env(self):
        cfg = config.load(
            Config,
            argv=[],
        )

        self.assertEqual(cfg.HOST, "env-host")
        self.assertEqual(cfg.PORT, 9001)

    @patch.dict(os.environ, {
        "APP_HOST": "localhost",
    })
    def test_env_prefix(self):
        cfg = config.load(
            Config,
            env_prefix="APP",
            argv=[],
        )

        self.assertEqual(cfg.HOST, "localhost")

    @patch.dict(os.environ, {
        "HOST": "localhost",
        "MODE": "prod",
    })
    def test_enum(self):
        cfg = config.load(
            Config,
            argv=[],
        )

        self.assertEqual(cfg.MODE, Mode.PROD)

    @patch.dict(os.environ, {
        "HOST": "localhost",
        "DEBUG": "true",
    })
    def test_bool(self):
        cfg = config.load(
            Config,
            argv=[],
        )

        self.assertTrue(cfg.DEBUG)

    def test_list_override(self):
        cfg = config.load(
            Config,
            overrides={
                "HOST": "localhost",
                "NUMBERS": [1, 2, 3],
            },
            argv=[],
        )

        self.assertEqual(cfg.NUMBERS, [1, 2, 3])

    @patch.dict(os.environ, {
        "HOST": "env",
    })
    def test_priority(self):
        cfg = config.load(
            Config,
            overrides={
                "HOST": "override",
            },
            priorities=(
                "envs",
                "overrides",
                "defaults",
            ),
            argv=[],
        )

        self.assertEqual(cfg.HOST, "env")

    def test_custom_source(self):

        def vault(name):
            if name == "HOST":
                return "vault-host"
            return None

        cfg = config.load(
            Config,
            extra_sources={
                "vault": vault,
            },
            priorities=(
                "vault",
                "defaults",
            ),
            argv=[],
        )

        self.assertEqual(cfg.HOST, "vault-host")

    def test_dump(self):
        cfg = config.load(
            Config,
            overrides={
                "HOST": "localhost",
            },
            argv=[],
        )

        self.assertEqual(
            config.dump(cfg),
            {
                "HOST": "localhost",
                "PORT": 8080,
                "DEBUG": False,
                "MODE": Mode.DEV,
                "NUMBERS": []
            },
        )

    def test_tuples(self):
        class Config:
            NUMBERS: tuple[int, ...]

        cfg = config.load(
            Config,
            overrides={
                "NUMBERS": "1,2,3",
            },
            argv=[],
        )

        self.assertEqual(cfg.NUMBERS, (1, 2, 3))
        self.assertIsInstance(cfg.NUMBERS, tuple)
        self.assertTrue(all(isinstance(x, int) for x in cfg.NUMBERS))

    def test_tuples_with_whitespace(self):
        class Config:
            NUMBERS: tuple[int, ...]

        cfg = config.load(
            Config,
            overrides={"NUMBERS": " 1,  2 , 3 "},
            argv=[],
        )

        self.assertEqual(cfg.NUMBERS, (1, 2, 3))

    def test_tuples_single_value(self):
        class Config:
            NUMBERS: tuple[int, ...]

        cfg = config.load(
            Config,
            overrides={"NUMBERS": "42"},
            argv=[],
        )

        self.assertEqual(cfg.NUMBERS, (42,))

    def test_tuples_empty(self):
        class Config:
            NUMBERS: tuple[int, ...]

        cfg = config.load(
            Config,
            overrides={"NUMBERS": ""},
            argv=[],
        )

        self.assertEqual(cfg.NUMBERS, tuple())

    def test_tuples_empty_values_are_ignored(self):
        class Config:
            NUMBERS: tuple[int, ...]

        cfg = config.load(
            Config,
            overrides={"NUMBERS": "1,,2, ,3"},
            argv=[],
        )

        self.assertEqual(cfg.NUMBERS, (1, 2, 3))

    def test_tuples_strings(self):
        class Config:
            VALUES: tuple[str, ...]

        cfg = config.load(
            Config,
            overrides={"VALUES": "foo,bar,baz"},
            argv=[],
        )

        self.assertEqual(cfg.VALUES, ("foo", "bar", "baz"))

    def test_tuples_bool(self):
        class Config:
            VALUES: tuple[bool, ...]

        cfg = config.load(
            Config,
            overrides={"VALUES": "true,false,yes"},
            argv=[],
        )

        self.assertEqual(cfg.VALUES, (True, False, True))

    def test_tuples_invalid_value(self):
        class Config:
            NUMBERS: tuple[int, ...]

        with self.assertRaises(ExceptionGroup):
            config.load(
                Config,
                overrides={"NUMBERS": "1,nope,3"},
                argv=[],
            )

    def test_tuples_multiple_invalid_values(self):
        class Config:
            NUMBERS: tuple[int, ...]

        try:
            config.load(
                Config,
                overrides={"NUMBERS": "nope,2,wat,4"},
                argv=[],
            )
        except ExceptionGroup as e:
            self.assertEqual(len(e.exceptions), 1)

            container_errors: ExceptionGroup = e.exceptions[0] # type:ignore

            self.assertIsInstance(container_errors, ExceptionGroup)
            self.assertEqual(len(container_errors.exceptions), 2)

        else:
            self.fail("Expected ExceptionGroup")

    def test_tuples_fixed_length(self):
        class Config:
            NUMBERS: tuple[int, int, int]

        cfg = config.load(
            Config,
            overrides={"NUMBERS": "1,2,3"},
            argv=[],
        )

        self.assertEqual(cfg.NUMBERS, (1, 2, 3))

    def test_tuples_fixed_length_too_few(self):
        class Config:
            NUMBERS: tuple[int, int, int]

        with self.assertRaises(Exception):
            config.load(
                Config,
                overrides={"NUMBERS": "1,2"},
                argv=[],
            )

    def test_tuples_fixed_length_too_many(self):
        class Config:
            NUMBERS: tuple[int, int, int]

        with self.assertRaises(Exception):
            config.load(
                Config,
                overrides={"NUMBERS": "1,2,3,4"},
                argv=[],
            )

    def test_tuples_mixed_types(self):
        class Config:
            VALUE: tuple[int, str, bool]

        cfg = config.load(
            Config,
            overrides={"VALUE": "42,hello,true"},
            argv=[],
        )

        self.assertEqual(cfg.VALUE, (42, "hello", True))
        self.assertIsInstance(cfg.VALUE[0], int)
        self.assertIsInstance(cfg.VALUE[1], str)
        self.assertIsInstance(cfg.VALUE[2], bool)
