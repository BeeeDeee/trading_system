"""Podpis proti oficiálním vektorům z Binance docs (docs/VERIFIED.md)."""

from arena.binance.signing import Ed25519Signer, HmacSigner


def test_hmac_futures_docs_vector():
    # USDⓈ-M General Info, HMAC SHA256 Example 1
    s = HmacSigner("k", "2b5eb11e18796d12d88f13dc27dbbd02c2cc51ff7059765ed9821957d82bb4d9")
    payload = ("symbol=BTCUSDT&side=BUY&type=LIMIT&quantity=1&price=9000&timeInForce=GTC"
               "&recvWindow=5000&timestamp=1591702613943")
    assert s.sign(payload) == "3c661234138461fcc7a7d8746c6558c9842d4e10870d2ecbedf7777cad694af9"


DOCS_ED25519_PEM = b"""-----BEGIN PRIVATE KEY-----
MC4CAQAwBQYDK2VwBCIEIIJEYWtGBrhACmb9Dvy+qa8WEf0lQOl1s4CLIAB9m89u
-----END PRIVATE KEY-----
"""


def test_ed25519_docs_vector():
    # Binance FIX API, Logon <A>: ilustrační klíč, payload a podpis z docs
    s = Ed25519Signer.from_pem("k", DOCS_ED25519_PEM)
    payload = "\x01".join(["A", "EXAMPLE", "SPOT", "1", "20240627-11:17:25.223"])
    assert s.sign(payload) == (
        "4MHXelVVcpkdwuLbl6n73HQUXUf1dse2PCgT1DYqW9w8AVZ1RACFGM+5UdlGPrQHrgtS3CvsRURC1oj73j8gCA=="
    )


def test_signers_do_not_leak_secrets():
    assert "2b5e" not in repr(HmacSigner("APIKEY", "2b5eb11e"))
    assert "APIKEY" not in repr(HmacSigner("APIKEY", "x"))
    assert "MC4C" not in repr(Ed25519Signer.from_pem("APIKEY", DOCS_ED25519_PEM))
