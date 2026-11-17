from src.features import FEATURE_NAMES, extract_features


def test_feature_count():
    assert len(extract_features("https://www.google.com")) == len(FEATURE_NAMES)


def test_ip_and_at():
    f = extract_features("http://192.168.1.5/paypal/login.php")
    assert f["has_ip_address"] == 1 and f["is_https"] == 0
    assert extract_features("http://paypal.com@evil.tk/")["has_at_symbol_trick"] == 1


def test_brand_misuse():
    assert extract_features("http://paypal.com.secure-login.xyz/verify")["brand_in_subdomain_or_path"] == 1
    assert extract_features("https://www.paypal.com/signin")["brand_in_subdomain_or_path"] == 0


def test_bare_domain_ok():
    assert extract_features("google.com")["is_https"] == 0
