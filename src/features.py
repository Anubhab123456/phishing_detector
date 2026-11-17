"""URL feature extraction (purely lexical - no network requests, so it is fast and safe)."""
import math
import re
from collections import Counter
from urllib.parse import urlparse, unquote

import numpy as np

SUSPICIOUS_KEYWORDS = [
    "login", "signin", "sign-in", "verify", "verification", "secure", "account",
    "update", "confirm", "banking", "password", "wallet", "billing", "support",
    "alert", "suspend", "unlock", "recover", "validate", "webscr", "authenticate",
    "security", "invoice", "payment", "limited",
]
BRANDS = [
    "paypal", "apple", "google", "microsoft", "amazon", "facebook", "netflix",
    "instagram", "linkedin", "dropbox", "adobe", "chase", "wellsfargo", "citibank",
    "hsbc", "icici", "sbi", "hdfc", "whatsapp", "outlook", "office365", "dhl", "fedex",
]
SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "rebrand.ly", "cutt.ly", "shorturl.at", "tiny.cc",
}
SUSPICIOUS_TLDS = {
    "xyz", "top", "tk", "ml", "ga", "cf", "gq", "click", "link", "work", "support",
    "live", "icu", "buzz", "country", "stream", "download", "zip", "mov", "rest",
}
_MULTI_TLDS = {"co.uk", "com.au", "co.in", "com.br", "co.jp", "org.uk", "ac.uk", "gov.in", "co.za", "ac.in"}

IP_RE = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")


def _split_host(host: str):
    """Return (subdomain, domain, suffix). Simple, dependency-free and offline."""
    if IP_RE.match(host):
        return "", host, ""
    parts = host.split(".")
    if len(parts) >= 3 and ".".join(parts[-2:]) in _MULTI_TLDS:
        return ".".join(parts[:-3]), parts[-3], ".".join(parts[-2:])
    if len(parts) >= 2:
        return ".".join(parts[:-2]), parts[-2], parts[-1]
    return "", host, ""


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def normalize_url(url: str) -> str:
    url = url.strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "http://" + url
    return url


FEATURE_NAMES = [
    "url_length", "hostname_length", "path_length", "query_length",
    "num_dots", "num_hyphens", "num_underscores", "num_slashes", "num_at",
    "num_question", "num_ampersand", "num_equals", "num_percent", "num_digits",
    "digit_ratio", "letter_ratio", "num_special_chars",
    "has_ip_address", "is_https", "num_subdomains", "suspicious_tld", "tld_length",
    "is_shortener", "num_suspicious_keywords", "hostname_entropy", "path_entropy",
    "has_punycode", "has_port", "double_slash_in_path", "brand_in_subdomain_or_path",
    "brand_in_domain_with_extras", "longest_token_length", "digits_in_hostname",
    "hyphens_in_hostname", "path_depth", "has_at_symbol_trick", "avg_token_length",
]

FEATURE_DESCRIPTIONS = {
    "url_length": "Total length of the URL",
    "hostname_length": "Length of the hostname",
    "path_length": "Length of the URL path",
    "query_length": "Length of the query string",
    "num_dots": "Number of '.' characters",
    "num_hyphens": "Number of '-' characters",
    "num_underscores": "Number of '_' characters",
    "num_slashes": "Number of '/' characters",
    "num_at": "Number of '@' characters",
    "num_question": "Number of '?' characters",
    "num_ampersand": "Number of '&' characters",
    "num_equals": "Number of '=' characters",
    "num_percent": "Number of '%' (encoded) characters",
    "num_digits": "Number of digits",
    "digit_ratio": "Share of characters that are digits",
    "letter_ratio": "Share of characters that are letters",
    "num_special_chars": "Number of non-alphanumeric characters",
    "has_ip_address": "Hostname is a raw IP address",
    "is_https": "Uses HTTPS",
    "num_subdomains": "Number of subdomain levels",
    "suspicious_tld": "TLD is commonly abused (e.g. .xyz, .tk)",
    "tld_length": "Length of the TLD",
    "is_shortener": "Known URL-shortening service",
    "num_suspicious_keywords": "Count of words like login/verify/secure",
    "hostname_entropy": "Randomness of the hostname",
    "path_entropy": "Randomness of the path",
    "has_punycode": "Hostname uses punycode (xn--)",
    "has_port": "Explicit non-standard port",
    "double_slash_in_path": "'//' appears inside the path",
    "brand_in_subdomain_or_path": "Brand name in subdomain/path, not the real domain",
    "brand_in_domain_with_extras": "Brand name embedded in a longer domain",
    "longest_token_length": "Longest alphanumeric token in the URL",
    "digits_in_hostname": "Digits in the hostname",
    "hyphens_in_hostname": "Hyphens in the hostname",
    "path_depth": "Directory depth of the path",
    "has_at_symbol_trick": "'@' used to hide the real host",
    "avg_token_length": "Average token length in the URL",
}


def _port(parsed):
    try:
        return parsed.port
    except ValueError:
        return None


def extract_features(url: str) -> dict:
    url = normalize_url(url)
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    path = unquote(parsed.path or "")
    query = parsed.query or ""
    full = url.lower()

    sub, domain, suffix = _split_host(host)
    is_ip = bool(IP_RE.match(host))
    tokens = [t for t in re.split(r"[^a-zA-Z0-9]+", full.split("://", 1)[-1]) if t]
    n = max(len(url), 1)
    digits = sum(c.isdigit() for c in url)
    letters = sum(c.isalpha() for c in url)
    registered = f"{domain}.{suffix}" if suffix else domain
    sub_and_path = f"{sub}.{path}".lower()
    port = _port(parsed)

    f = {
        "url_length": len(url),
        "hostname_length": len(host),
        "path_length": len(path),
        "query_length": len(query),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_underscores": url.count("_"),
        "num_slashes": url.count("/"),
        "num_at": url.count("@"),
        "num_question": url.count("?"),
        "num_ampersand": url.count("&"),
        "num_equals": url.count("="),
        "num_percent": url.count("%"),
        "num_digits": digits,
        "digit_ratio": digits / n,
        "letter_ratio": letters / n,
        "num_special_chars": sum(not c.isalnum() for c in url),
        "has_ip_address": int(is_ip),
        "is_https": int(parsed.scheme == "https"),
        "num_subdomains": 0 if (is_ip or not sub) else len(sub.split(".")),
        "suspicious_tld": int(suffix.split(".")[-1] in SUSPICIOUS_TLDS),
        "tld_length": len(suffix),
        "is_shortener": int(registered in SHORTENERS),
        "num_suspicious_keywords": sum(full.count(k) for k in SUSPICIOUS_KEYWORDS),
        "hostname_entropy": _entropy(host),
        "path_entropy": _entropy(path),
        "has_punycode": int("xn--" in host),
        "has_port": int(port not in (None, 80, 443)),
        "double_slash_in_path": int("//" in path),
        "brand_in_subdomain_or_path": int(any(b in sub_and_path and b not in domain for b in BRANDS)),
        "brand_in_domain_with_extras": int(any(b in domain and domain != b for b in BRANDS)),
        "longest_token_length": max((len(t) for t in tokens), default=0),
        "digits_in_hostname": sum(c.isdigit() for c in host),
        "hyphens_in_hostname": host.count("-"),
        "path_depth": len([p for p in path.split("/") if p]),
        "has_at_symbol_trick": int("@" in (parsed.netloc or "")),
        "avg_token_length": float(np.mean([len(t) for t in tokens])) if tokens else 0.0,
    }
    return {k: f[k] for k in FEATURE_NAMES}


def features_to_vector(url: str) -> np.ndarray:
    f = extract_features(url)
    return np.array([f[k] for k in FEATURE_NAMES], dtype=float)
