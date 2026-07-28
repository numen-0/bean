from tests.utils import BaseTest
from bean import config


class TestNormalizers(BaseTest):

    def test_instancemethod(self):
        class Config:
            HOST: str

            @config.normalizer("HOST")
            def lower(self, host: str) -> str:
                return host.lower()


        cfg = config.load(
            Config,
            overrides={"HOST": "LOCALHOST"},
            argv=[],
        )

        self.assertEqual(cfg.HOST, "localhost")

    def test_staticmethod(self):
        class Config:
            HOST: str

            @staticmethod
            @config.normalizer("HOST")
            def lower(host: str) -> str:
                return host.lower()

        cfg = config.load(
            Config,
            overrides={"HOST": "LOCALHOST"},
            argv=[],
        )

        self.assertEqual(cfg.HOST, "localhost")

    def test_classmethod(self):
        class Config:
            HOST: str

            @classmethod
            @config.normalizer("HOST")
            def lower(cls, host: str) -> str:
                return host.lower()

        cfg = config.load(
            Config,
            overrides={"HOST": "LOCALHOST"},
            argv=[],
        )

        self.assertEqual(cfg.HOST, "localhost")

    def test_any(self):
        class Config:
            HOST: str

            @config.normalizer()
            def normalize(self) -> None:
                self.HOST = self.HOST.lower()

        cfg = config.load(
            Config,
            overrides={"HOST": "LOCALHOST"},
            argv=[],
        )

        self.assertEqual(cfg.HOST, "localhost")

    def test_multiple(self):
        class Config:
            HOST: str
            PORT: int

            @config.normalizer("HOST", "PORT")
            def normalize(self, host: str, port: int) -> tuple[str, int]:
                return host.lower(), port + 1

        cfg = config.load(
            Config,
            overrides={
                "HOST": "LOCALHOST",
                "PORT": 8080,
            },
            argv=[],
        )

        self.assertEqual(cfg.HOST, "localhost")
        self.assertEqual(cfg.PORT, 8081)

    def test_serial(self):
        class Config:
            HOST: str
            NAME: str

            @config.normalizer("HOST", "NAME", series=True)
            def lower(self, value: str) -> str:
                return value.lower()

        cfg = config.load(
            Config,
            overrides={
                "HOST": "LOCALHOST",
                "NAME": "ADMIN",
            },
            argv=[],
        )

        self.assertEqual(cfg.HOST, "localhost")
        self.assertEqual(cfg.NAME, "admin")

    def test_disallow_overlap(self):
        class Config:
            HOST: str

            @config.normalizer("HOST")
            def lower(self, host):
                return host.lower()

            @config.normalizer("HOST")
            def upper(self, host):
                return host.upper()

        cfg = config.build(
            Config,
            overrides={"HOST": "LocalHost"},
            argv=[],
        )

        with self.assertRaises(KeyError):
            config.normalize(cfg, allow_overlap=False)

    def test_allow_overlap(self):
        class Config:
            HOST: str

            @config.normalizer("HOST", priority=10)
            def lower(self, host):
                return host.lower()

            @config.normalizer("HOST", priority=20)
            def suffix(self, host):
                return host + ".local"

        cfg = config.build(
            Config,
            overrides={"HOST": "LOCALHOST"},
            argv=[],
        )

        cfg = config.normalize(
            cfg,
            allow_overlap=True,
        )

        self.assertEqual(cfg.HOST, "localhost.local")

    def test_priority(self):
        order = []

        class Config:
            DEBUG: bool = True

            @config.validator(priority=30)
            def c(self):
                order.append(3)

            @config.validator(priority=20)
            def b(self):
                order.append(2)

            @config.validator(priority=20)
            def a(self):
                order.append(1)

            @config.validator(priority=40)
            def d(self):
                order.append(4)

        config.load(Config, argv=[])

        self.assertEqual(order, [1, 2, 3, 4])

    def test_unknown_field(self):
        class Config:
            HOST: str

            @config.normalizer("DEBUG")
            def lower(self, host):
                return host.lower()

        cfg = config.build(
            Config,
            overrides={"HOST": "LocalHost"},
            argv=[],
        )

        with self.assertRaises(KeyError):
            config.normalize(cfg)

    def test_parameter_count_mismatch(self):
        class Config:
            DEBUG: bool = True
            HOST: str
            PORT: int

            @config.normalizer("HOST", "PORT", "DEBUG")
            def normalize(self, host: str, port: int) -> tuple[str, int, bool]:
                return host.lower(), port + 1, True

        cfg = config.build(
            Config,
            overrides={
                "HOST": "LOCALHOST",
                "PORT": 8080,
            },
            argv=[],
        )

        with self.assertRaises(TypeError):
            config.normalize(cfg)

    def test_return_value_count_mismatch(self):
        class Config:
            DEBUG: bool = True
            HOST: str
            PORT: int

            @config.normalizer("HOST", "PORT", "DEBUG")
            def normalize(self, host: str, port: int, _) -> tuple[str, int]:
                return host.lower(), port + 1

        cfg = config.build(
            Config,
            overrides={
                "HOST": "LOCALHOST",
                "PORT": 8080,
            },
            argv=[],
        )

        with self.assertRaises(ValueError):
            config.normalize(cfg)

    def test_series_requires_fields(self):
        class Config:
            DEBUG: bool = True

            @config.normalizer(series=True)
            def normalize(self, value):
                return value

        cfg = config.build(Config, argv=[])

        with self.assertRaises(ValueError):
            config.normalize(cfg)

    def test_empty_normalizer_returns_value(self):
        class Config:
            DEBUG: bool = True

            @config.normalizer()
            def normalize(self):
                return 123

        cfg = config.build(Config, argv=[])

        with self.assertRaises(TypeError):
            config.normalize(cfg)

    def test_series_returns_none(self):
        class Config:
            HOST: str

            @config.normalizer("HOST", series=True)
            def normalize(self, _):
                return None

        cfg = config.build(
            Config,
            overrides={"HOST": "LOCALHOST"},
            argv=[],
        )

        with self.assertRaises(TypeError):
            config.normalize(cfg)

