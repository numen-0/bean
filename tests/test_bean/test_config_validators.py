from tests.utils import BaseTest
from bean.core import (
    dirExists, fileExists, isDate, isEmail, isHost, isIPv4, isIPv6,
    isNegative, isPort, isPositive, isUrl, nonEmpty, pathExists,
)

class TestConfigValidators(BaseTest):

    # text

    def test_non_empty(self):
        cases = [
            ("", False),
            ("   ", False),
            ("hello", True),
            ("hi", True),
        ]
        self.assertCases(cases, nonEmpty)

    def test_is_date(self):
        cases = [
            ("2024-01-01", True),
            ("1999-12-31", True),
            ("2024-13-01", False),
            ("not-a-date", False),
        ]
        self.assertCases(cases, isDate())

    def test_is_email(self):
        cases = [
            ("user@example.com", True),
            ("cool.user@example.com", True),
            ("bad-email", False),
            ("user@", False),
            ("@example.com", False),
        ]
        self.assertCases(cases, isEmail)

    # numbers
    
    def test_is_positive(self):
        cases = [
            (1, True),
            (0, False),
            (-1, False),
            (-0.2, False),
        ]
        self.assertCases(cases, isPositive)

    def test_is_negative(self):
        cases = [
            (-1, True),
            (0, False),
            (5, False),
            (0.2, False),
        ]
        self.assertCases(cases, isNegative)

    def test_is_port(self):
        cases = [
            (80, True),
            (0, False),
            (65535, True),
            (65536, False),
            (-1, False),
            (8080, True),
        ]
        self.assertCases(cases, isPort)

    # network

    def test_ipv4(self):
        cases = [
            ("127.0.0.1", True),
            ("192.168.1.1", True),
            ("256.0.0.1", False),
            ("abc.def.ghi.jkl", False),
        ]
        self.assertCases(cases, isIPv4)

    def test_ipv6(self):
        cases = [
            ("::1", True),
            ("2001:db8::1", True),
            ("invalid::ip", False),
        ]
        self.assertCases(cases, isIPv6)

    def test_is_host(self):
        cases = [
            ("localhost", True),
            ("example.com", True),
            ("google.com", True),
            ("packaging.python.org", True),
            ("invalid_host!", False),
        ]
        self.assertCases(cases, isHost)

    def test_is_url(self):
        cases = [
            ("http://example.com", True),
            ("https://example.com/path", True),
            ("ftp://example.com", True),
            ("not-a-url", False),
        ]
        self.assertCases(cases, isUrl)

    # path

    def test_path_exists(self):
        cases = [
            (".", True),
            ("this_path_should_not_exist_123", False),
        ]
        self.assertCases(cases, pathExists)

    def test_dir_exists(self):
        cases = [
            (".", True),
            ("/home", True),
            ("this_path_should_not_exist_123", False),
        ]
        self.assertCases(cases, dirExists)

    def test_file_exists(self):
        cases = [
            ("./README.md", True),
            ("this_path_should_not_exist_123", False),
        ]
        self.assertCases(cases, fileExists)
