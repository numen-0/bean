from tests.utils import BaseTest
from bean import config


class TestValidators(BaseTest):

    def test_instancemethod(self):
        class Config:
            PORT: int

            @config.validator("PORT")
            def valid(self, port: int) -> bool:
                return port > 0

        cfg = config.build(
            Config,
            overrides={"PORT": 8080},
            argv=[],
        )

        config.validate(cfg)

    def test_staticmethod(self):
        class Config:
            PORT: int

            @staticmethod
            @config.validator("PORT")
            def valid(port: int) -> bool:
                return port > 0

        cfg = config.build(
            Config,
            overrides={"PORT": 8080},
            argv=[],
        )

        config.validate(cfg)

    def test_classmethod(self):
        class Config:
            PORT: int

            @classmethod
            @config.validator("PORT")
            def valid(cls, port: int) -> bool:
                return port > 0

        cfg = config.build(
            Config,
            overrides={"PORT": 8080},
            argv=[],
        )

        config.validate(cfg)

    def test_any(self):
        calls = []

        class Config:
            DEBUG: bool = True

            @config.validator()
            def valid(self) -> bool:
                calls.append(1)
                return self.DEBUG

        cfg = config.build(Config, argv=[])

        config.validate(cfg)

        self.assertEqual(calls, [1])

    def test_multiple(self):
        class Config:
            HOST: str
            PORT: int

            @config.validator("HOST", "PORT")
            def valid(self, host: str, port: int) -> bool:
                return host == "localhost" and port == 8080

        cfg = config.build(
            Config,
            overrides={
                "HOST": "localhost",
                "PORT": 8080,
            },
            argv=[],
        )

        config.validate(cfg)

    def test_serial(self):
        class Config:
            A: int
            B: int
            C: int

            @config.validator("A", "B", "C", series=True)
            def positive(self, value: int) -> bool:
                return value > 0

        cfg = config.build(
            Config,
            overrides={
                "A": 1,
                "B": 2,
                "C": 3,
            },
            argv=[],
        )

        config.validate(cfg)

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

        cfg = config.build(Config, argv=[])

        config.validate(cfg)

        self.assertEqual(order, [1, 2, 3, 4])

    def test_unknown_field(self):
        class Config:
            DEBUG: bool = True

            @config.validator("HOST")
            def valid(self, _):
                return True

        cfg = config.build(Config, argv=[])

        with self.assertRaises(KeyError):
            config.validate(cfg)

    def test_parameter_count_mismatch(self):
        class Config:
            HOST: str
            PORT: int
            DEBUG: bool

            @config.validator("HOST", "PORT", "DEBUG")
            def valid(self, host: str, port: int):
                _ = (host, port)
                return True

        cfg = config.build(
            Config,
            overrides={
                "HOST": "localhost",
                "PORT": 8080,
                "DEBUG": True,
            },
            argv=[],
        )

        with self.assertRaises(ExceptionGroup):
            config.validate(cfg)

    def test_series_requires_fields(self):
        class Config:
            DEBUG: bool = True

            @config.validator(series=True)
            def valid(self, _):
                return True

        cfg = config.build(Config, argv=[])

        with self.assertRaises(ExceptionGroup):
            config.validate(cfg)

    def test_returns_false(self):
        class Config:
            DEBUG: bool = True

            @config.validator()
            def valid(self):
                return False

        cfg = config.build(Config, argv=[])

        with self.assertRaises(ExceptionGroup):
            config.validate(cfg)

    def test_returns_none(self):
        class Config:
            DEBUG: bool = True

            @config.validator("DEBUG")
            def valid(self, _):
                return None

        cfg = config.build(Config, argv=[])

        config.validate(cfg)

    def test_short_circuit(self):
        order = []

        class Config:
            DEBUG: bool = True

            @config.validator(priority=10)
            def first(self):
                order.append(1)
                return False

            @config.validator(priority=20)
            def second(self):
                order.append(2)
                return True

        cfg = config.build(Config, argv=[])

        with self.assertRaises(ExceptionGroup):
            config.validate(cfg, short_circuit=True)

        self.assertEqual(order, [1])

    def test_no_short_circuit(self):
        order = []

        class Config:
            DEBUG: bool = True

            @config.validator(priority=10)
            def first(self):
                order.append(1)
                return False

            @config.validator(priority=20)
            def second(self):
                order.append(2)
                return True

        cfg = config.build(Config, argv=[])

        with self.assertRaises(ExceptionGroup):
            config.validate(cfg)

        self.assertEqual(order, [1, 2])

    def test_is_valid_true(self):
        class Config:
            DEBUG: bool = True

            @config.validator()
            def valid(self):
                return True

        cfg = config.build(Config, argv=[])

        self.assertTrue(config.is_valid(cfg))

    def test_is_valid_false(self):
        class Config:
            DEBUG: bool = True

            @config.validator()
            def valid(self):
                return False

        cfg = config.build(Config, argv=[])

        self.assertFalse(config.is_valid(cfg))

