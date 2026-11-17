"""Command-line classifier:  python cli.py https://example.com [more urls...]"""
import sys

from src.predict import PhishingDetector

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python cli.py <url> [url ...]")
        sys.exit(1)
    det = PhishingDetector()
    for u in sys.argv[1:]:
        r = det.predict(u)
        print(f"\n{u}\n  -> {r['label'].upper()}  (confidence {r['confidence']:.1%}, "
              f"P(phishing)={r['phishing_probability']:.1%})")
        print("  Top contributing features:")
        for k, v in list(r["contributions"].items())[:5]:
            print(f"    {k:30s} {v:+.3f}  (value={r['features'][k]:.2f})")
