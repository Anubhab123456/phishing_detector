"""Synthetic URL dataset generator.

NOTE: This produces *realistic-looking* URLs so the project runs fully offline.
For production-grade accuracy, train on a real dataset (see README):
    python -m src.train --csv path/to/data.csv
CSV format: columns `url` and `label` (1 = phishing, 0 = legitimate).
"""
import random
import string

import pandas as pd

LEGIT_DOMAINS = [
    "google.com", "youtube.com", "wikipedia.org", "github.com", "stackoverflow.com",
    "amazon.com", "microsoft.com", "apple.com", "paypal.com", "netflix.com",
    "linkedin.com", "twitter.com", "reddit.com", "bbc.co.uk", "nytimes.com",
    "cnn.com", "medium.com", "dropbox.com", "adobe.com", "zoom.us", "slack.com",
    "python.org", "mozilla.org", "cloudflare.com", "stripe.com", "shopify.com",
    "ebay.com", "walmart.com", "target.com", "imdb.com", "spotify.com", "nasa.gov",
    "who.int", "mit.edu", "stanford.edu", "iitb.ac.in", "irctc.co.in", "sbi.co.in",
    "hdfcbank.com", "icicibank.com", "flipkart.com", "swiggy.com", "zomato.com",
    "indiatimes.com", "thehindu.com", "coursera.org", "udemy.com", "kaggle.com",
    "huggingface.co", "docker.com", "kubernetes.io", "npmjs.com", "pypi.org",
]
WORDS = ["blue", "green", "fast", "smart", "cloud", "data", "bright", "north", "river",
         "stone", "pixel", "urban", "swift", "maple", "orbit", "nova", "delta", "lotus",
         "craft", "bloom", "habit", "trail", "metro", "forge", "atlas", "pine", "cedar"]
NOUNS = ["shop", "labs", "studio", "news", "tech", "foods", "travel", "books", "works",
         "media", "health", "games", "design", "store", "hub", "systems"]
LEGIT_TLDS = ["com", "org", "net", "io", "co", "edu", "in", "co.uk", "dev", "app"]
LEGIT_PATHS = ["", "/", "/about", "/contact", "/products", "/blog", "/docs", "/pricing",
               "/help/articles", "/careers", "/news/2024/05/story", "/search", "/learn/python",
               "/account/settings", "/support", "/login", "/signin", "/user/profile",
               "/wiki/Machine_learning", "/products/item-4821", "/category/books/page/2"]
LEGIT_QUERIES = ["", "", "", "?q=python+tutorial", "?page=2&sort=asc", "?ref=homepage",
                 "?utm_source=newsletter&utm_medium=email", "?id=1042&lang=en"]
BRANDS = ["paypal", "apple", "google", "microsoft", "amazon", "facebook", "netflix",
          "instagram", "linkedin", "dropbox", "adobe", "chase", "wellsfargo", "hsbc",
          "icici", "sbi", "hdfc", "whatsapp", "outlook", "office365", "dhl", "fedex"]
KEYS = ["login", "signin", "verify", "secure", "account", "update", "confirm", "banking",
        "password", "wallet", "billing", "support", "alert", "unlock", "recover", "validate"]
BAD_TLDS = ["xyz", "top", "tk", "ml", "ga", "cf", "gq", "click", "link", "work", "live",
            "icu", "buzz", "info", "online", "site", "cc", "ru", "biz"]
PHISH_FILES = ["index.php", "login.php", "signin.html", "verify.html", "secure.php",
               "update.aspx", "confirm.php", "account.html", ""]


def _rand(n, chars=string.ascii_lowercase + string.digits):
    return "".join(random.choices(chars, k=n))


def _legit():
    if random.random() < 0.5:
        domain = random.choice(LEGIT_DOMAINS)
    else:
        domain = f"{random.choice(WORDS)}{random.choice(['', '-'])}{random.choice(NOUNS)}.{random.choice(LEGIT_TLDS)}"
    sub = random.choices(["www.", "", "blog.", "docs.", "support.", "shop.", "mail.", "en."],
                         weights=[55, 20, 6, 6, 5, 4, 2, 2])[0]
    scheme = "https" if random.random() < 0.93 else "http"
    return f"{scheme}://{sub}{domain}{random.choice(LEGIT_PATHS)}{random.choice(LEGIT_QUERIES)}"


def _phish():
    brand = random.choice(BRANDS)
    key1, key2 = random.choice(KEYS), random.choice(KEYS)
    tld = random.choice(BAD_TLDS)
    scheme = "https" if random.random() < 0.45 else "http"  # phishing sites use HTTPS too
    fname = random.choice(PHISH_FILES)
    kind = random.randrange(10)
    if kind == 0:   # brand-keyword hyphen domain
        return f"{scheme}://{brand}-{key1}-{key2}.{tld}/{key1}/{fname}"
    if kind == 1:   # brand as subdomain of malicious domain
        return f"{scheme}://{brand}.com.{key1}-{_rand(random.randint(4, 9))}.{tld}/{key2}/{fname}"
    if kind == 2:   # raw IP
        ip = ".".join(str(random.randint(1, 254)) for _ in range(4))
        return f"http://{ip}/{brand}/{key1}/{fname}"
    if kind == 3:   # @ trick
        return f"http://{brand}.com@{_rand(random.randint(6, 12))}.{tld}/{key1}"
    if kind == 4:   # deep subdomain chain
        subs = ".".join(random.choices(KEYS + [brand], k=random.randint(3, 5)))
        return f"{scheme}://{subs}.{_rand(8)}.{tld}/{fname}"
    if kind == 5:   # brand in path on compromised host
        host = f"{random.choice(WORDS)}{random.choice(NOUNS)}{random.randint(1, 999)}.{random.choice(['com', 'net', 'ru', 'in'])}"
        return f"http://{host}/wp-includes/{brand}/{key1}/{fname}"
    if kind == 6:   # gibberish domain with long path
        return f"{scheme}://{_rand(random.randint(12, 22))}.{tld}/{key1}{random.choice(['', '-'])}{key2}/{_rand(10)}/{fname}"
    if kind == 7:   # punycode / encoded
        return f"{scheme}://xn--{_rand(8)}-{_rand(4)}.{tld}/{key1}%2F{key2}?id={_rand(16)}&token={_rand(24)}"
    if kind == 8:   # shortener
        return f"http://{random.choice(['bit.ly', 'tinyurl.com', 'cutt.ly', 'is.gd'])}/{_rand(random.randint(5, 8), string.ascii_letters + string.digits)}"
    # kind 9: subtle - clean-ish domain, suspicious path/query
    host = f"{random.choice(WORDS)}{random.choice(NOUNS)}.{random.choice(['com', 'net', 'online', 'site'])}"
    return f"{scheme}://{host}/{key1}/{brand}/{fname}?session={_rand(20)}&redirect={key2}"


def _hard_legit():
    """Legitimate URLs that *look* suspicious (long, keyword-heavy, IPs, odd TLDs)."""
    r = random.randrange(6)
    host = f"{random.choice(WORDS)}-{random.choice(NOUNS)}-{random.choice(WORDS)}.{random.choice(['com', 'org', 'xyz', 'online', 'site', 'info', 'live'])}"
    if r == 0:
        return f"https://{host}/account/{random.choice(KEYS)}?id={_rand(12)}&session={_rand(16)}"
    if r == 1:
        return f"http://{random.randint(10, 192)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}:8080/admin/{random.choice(KEYS)}"
    if r == 2:
        return f"https://secure.{random.choice(LEGIT_DOMAINS)}/{random.choice(KEYS)}/{_rand(8)}?redirect=%2Fhome&token={_rand(20)}"
    if r == 3:
        return f"https://{random.choice(['accounts', 'login', 'signin', 'verify'])}.{random.choice(LEGIT_DOMAINS)}/{random.choice(KEYS)}"
    if r == 4:
        return f"https://bit.ly/{_rand(6, string.ascii_letters + string.digits)}"
    return f"https://{host}/products/{random.choice(WORDS)}-{random.choice(NOUNS)}-{random.randint(1, 9999)}"


def _hard_phish():
    """Phishing URLs that look clean (short, https, plausible domain)."""
    r = random.randrange(3)
    host = f"{random.choice(WORDS)}{random.choice(NOUNS)}.{random.choice(['com', 'net', 'org', 'co'])}"
    if r == 0:
        return f"https://{host}/{random.choice(KEYS)}"
    if r == 1:
        return f"https://www.{host}/{random.choice(['index.html', 'home.php', 'portal/auth', 'secure/index.html'])}"
    return f"https://{random.choice(WORDS)}-{random.choice(NOUNS)}.{random.choice(['com', 'net'])}/{random.choice(KEYS)}/{_rand(6)}"


def generate_dataset(n_per_class: int = 6000, seed: int = 42, hard_frac: float = 0.25,
                     label_noise: float = 0.03) -> pd.DataFrame:
    random.seed(seed)
    rows = []
    for _ in range(n_per_class):
        rows.append((_hard_legit() if random.random() < hard_frac else _legit(), 0))
        rows.append((_hard_phish() if random.random() < hard_frac else _phish(), 1))
    # label noise mimics real-world mislabelled samples
    rows = [(u, 1 - y if random.random() < label_noise else y) for u, y in rows]
    random.shuffle(rows)
    return pd.DataFrame(rows, columns=["url", "label"]).drop_duplicates("url").reset_index(drop=True)


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("data/urls.csv", index=False)
    print(f"Saved {len(df)} URLs to data/urls.csv")
    print(df.sample(8, random_state=1).to_string())
