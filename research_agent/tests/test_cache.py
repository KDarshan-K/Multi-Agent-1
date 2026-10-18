import time
from tools import cache


def test_cache_hit_and_miss():
    cache.clear_cache()
    calls = 0

    def mock_fetch():
        nonlocal calls
        calls += 1
        return {"data": 123}

    res1 = cache.cached_call("test_key", 10, mock_fetch)
    res2 = cache.cached_call("test_key", 10, mock_fetch)

    assert res1 == {"data": 123}
    assert res2 == {"data": 123}
    assert calls == 1


def test_cache_expiration():
    cache.clear_cache()
    calls = 0

    def mock_fetch():
        nonlocal calls
        calls += 1
        return "val"

    res1 = cache.cached_call("exp_key", 0.05, mock_fetch)
    assert res1 == "val"
    assert calls == 1

    time.sleep(0.1)
    res2 = cache.cached_call("exp_key", 0.05, mock_fetch)
    assert res2 == "val"
    assert calls == 2
