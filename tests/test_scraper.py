"""Tests for the scraper module."""

from bs4 import BeautifulSoup

from src.scraper.sources import ALL_SITES, DEFAULT_SITES, PUT_THIS_ON, SiteConfig


def test_site_config_fields():
    """Verify all site configs have required fields."""
    for name, config in ALL_SITES.items():
        assert isinstance(config, SiteConfig)
        assert config.name == name
        assert config.base_url.startswith("https://") or config.base_url.startswith("https://")
        assert len(config.listing_urls) > 0
        assert config.article_link_selector
        assert config.title_selector
        assert config.body_selector
        assert config.request_delay >= 1.0  # Minimum politeness


def test_default_sites_exist():
    """Check that all default sites are valid."""
    for site in DEFAULT_SITES:
        assert site in ALL_SITES


def test_putthison_config():
    """Spot check Put This On config values."""
    assert PUT_THIS_ON.name == "putthison"
    assert "putthison.com" in PUT_THIS_ON.base_url
    assert PUT_THIS_ON.request_delay >= 2.0


def test_no_restricted_sites():
    """Verify restricted sites are not in the site list."""
    assert "gq" not in ALL_SITES
    assert "highsnobiety" not in ALL_SITES
    assert "thezoereport" not in ALL_SITES  # JS-rendered, can't scrape


def test_extract_article_links_filters_same_domain():
    """Test that link extraction only keeps same-domain links."""
    from src.scraper.scrape import extract_article_links

    html = """
    <html><body>
        <a href="/some-article-about-blazers">Blazer Guide</a>
        <a href="https://putthison.com/how-to-wear-a-suit">Suit Guide</a>
        <a href="https://otherdomain.com/article">External</a>
    </body></html>
    """
    soup = BeautifulSoup(html, "lxml")
    links = extract_article_links(soup, PUT_THIS_ON)
    for link in links:
        assert "putthison.com" in link
