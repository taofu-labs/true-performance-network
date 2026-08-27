import pytest

from common.urls import InvalidBaseUrl, validate_base_url


@pytest.mark.parametrize("url,expected", [
    ("https://val0.trueperformancenetwork.com", "https://val0.trueperformancenetwork.com"),
    ("https://val0.trueperformancenetwork.com/", "https://val0.trueperformancenetwork.com"),
    ("http://localhost:9200", "http://localhost:9200"),
    ("http://localhost:9200//", "http://localhost:9200"),
    ("  https://val0.example.com  ", "https://val0.example.com"),
    ("https://example.com/tpn", "https://example.com/tpn"),
])
def test_accepts_valid_urls(url, expected):
    assert validate_base_url(url) == expected


def test_rejects_doubled_scheme():
    """The typo this guard exists for — it shipped to mainnet once."""
    with pytest.raises(InvalidBaseUrl) as e:
        validate_base_url("https://https://val0.trueperformancenetwork.com")
    assert "doubled scheme" in str(e.value)


@pytest.mark.parametrize("url", [
    "",
    "   ",
    "val0.trueperformancenetwork.com",   # no scheme
    "ftp://val0.example.com",            # wrong scheme
    "https://",                          # no host
    "https:///path",                     # no host, path only
])
def test_rejects_malformed_urls(url):
    with pytest.raises(InvalidBaseUrl):
        validate_base_url(url)


def test_error_names_the_setting():
    with pytest.raises(InvalidBaseUrl) as e:
        validate_base_url("nonsense", setting_name="LEADER_VALIDATOR_URL")
    assert "LEADER_VALIDATOR_URL" in str(e.value)
