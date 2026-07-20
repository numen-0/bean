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

    def test_dump_meta(self):
        cfg = config.load(
            Config,
            overrides={
                "HOST": "localhost",
            },
            argv=[],
        )

        meta = config.dump_meta(cfg)

        self.assertEqual(meta["HOST"][0], "localhost")
        self.assertEqual(meta["HOST"][1], str)
        self.assertEqual(meta["HOST"][2], "overrides")

