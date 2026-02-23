"""
Web scraper for fashion editorial articles.

Scrapes articles from configured fashion sites, extracts structured content,
and saves them as JSON files. Respects robots.txt and rate limits.
"""

import json
import logging
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

from src.scraper.sources import ALL_SITES, DEFAULT_SITES, SiteConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# Cache parsed robots.txt per domain to avoid re-fetching
_robots_cache: dict[str, RobotFileParser] = {}


def check_robots_txt(base_url: str, path: str) -> bool:
    """Check if a URL is allowed by the site's robots.txt.

    Fetches robots.txt using our custom headers (not urllib's default UA)
    to avoid false 403 blocks from CDNs like Cloudflare.
    """
    if base_url in _robots_cache:
        return _robots_cache[base_url].can_fetch("*", urljoin(base_url, path))

    robots_url = urljoin(base_url, "/robots.txt")
    rp = RobotFileParser()
    rp.set_url(robots_url)

    try:
        resp = requests.get(robots_url, headers=HEADERS, timeout=10)
        if resp.status_code == 200:
            rp.parse(resp.text.splitlines())
        else:
            # If robots.txt is missing or inaccessible, allow everything
            logger.warning(f"robots.txt returned {resp.status_code} for {base_url}, allowing all")
            rp.parse([])  # Empty rules = allow all
    except requests.RequestException:
        logger.warning(f"Could not fetch robots.txt for {base_url}, allowing all")
        rp.parse([])

    _robots_cache[base_url] = rp
    return rp.can_fetch("*", urljoin(base_url, path))


def fetch_page(url: str, delay: float = 2.0) -> BeautifulSoup | None:
    """Fetch a page and return parsed BeautifulSoup, or None on failure."""
    try:
        time.sleep(delay)
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        return BeautifulSoup(response.text, "lxml")
    except requests.RequestException as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return None


def extract_article_links(soup: BeautifulSoup, config: SiteConfig) -> list[str]:
    """Extract article URLs from a listing page."""
    links = []
    for tag in soup.select(config.article_link_selector):
        href = tag.get("href")
        if not href:
            continue
        full_url = urljoin(config.base_url, href)
        # Only keep links from the same domain
        if urlparse(full_url).netloc == urlparse(config.base_url).netloc:
            links.append(full_url)
    return list(set(links))


def extract_article(url: str, soup: BeautifulSoup, config: SiteConfig) -> dict | None:
    """Extract structured article data from a page."""
    title_el = soup.select_one(config.title_selector)
    body_el = soup.select_one(config.body_selector)

    if not title_el or not body_el:
        logger.warning(f"Missing title or body for {url}")
        return None

    # Remove unwanted elements from body
    for selector in config.remove_selectors:
        for el in body_el.select(selector):
            el.decompose()

    # Extract text, preserving paragraph structure
    paragraphs = []
    if config.content_selectors:
        # Use site-specific content selectors (e.g. Refinery29's .section-text divs)
        for selector in config.content_selectors:
            for el in body_el.select(selector):
                text = el.get_text(strip=True)
                if text and len(text) > 20:
                    paragraphs.append(text)
    else:
        for p in body_el.find_all(["p", "h2", "h3", "li"]):
            text = p.get_text(strip=True)
            if text and len(text) > 20:
                paragraphs.append(text)

    body_text = "\n\n".join(paragraphs)

    if len(body_text) < 200:
        logger.warning(f"Article too short ({len(body_text)} chars), skipping: {url}")
        return None

    # Extract metadata
    author = ""
    if config.author_selector:
        author_el = soup.select_one(config.author_selector)
        if author_el:
            author = author_el.get_text(strip=True)

    date = ""
    if config.date_selector:
        date_el = soup.select_one(config.date_selector)
        if date_el:
            date = date_el.get("datetime", date_el.get_text(strip=True))

    category = ""
    if config.category_selector:
        cat_el = soup.select_one(config.category_selector)
        if cat_el:
            category = cat_el.get_text(strip=True)

    return {
        "title": title_el.get_text(strip=True),
        "url": url,
        "source": config.name,
        "author": author,
        "date": date,
        "category": category,
        "body_text": body_text,
    }


def scrape_site(config: SiteConfig, max_articles: int = 50) -> list[dict]:
    """Scrape articles from a single site."""
    logger.info(f"Scraping {config.name} (max {max_articles} articles)...")

    # Collect article links from listing pages
    all_links = []
    for listing_url in config.listing_urls:
        parsed = urlparse(listing_url)
        if not check_robots_txt(config.base_url, parsed.path):
            logger.warning(f"Blocked by robots.txt: {listing_url}")
            continue

        soup = fetch_page(listing_url, delay=config.request_delay)
        if soup:
            links = extract_article_links(soup, config)
            all_links.extend(links)
            logger.info(f"  Found {len(links)} links on {listing_url}")

    all_links = list(set(all_links))[:max_articles]
    logger.info(f"  Total unique links to scrape: {len(all_links)}")

    # Scrape each article
    articles = []
    for url in all_links:
        parsed = urlparse(url)
        if not check_robots_txt(config.base_url, parsed.path):
            logger.warning(f"Blocked by robots.txt: {url}")
            continue

        soup = fetch_page(url, delay=config.request_delay)
        if not soup:
            continue

        article = extract_article(url, soup, config)
        if article:
            articles.append(article)
            logger.info(f"  Scraped: {article['title'][:60]}...")

    logger.info(f"  Successfully scraped {len(articles)} articles from {config.name}")
    return articles


def save_articles(articles: list[dict], site_name: str) -> Path:
    """Save scraped articles to a JSON file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / f"{site_name}_articles.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(articles)} articles to {output_path}")
    return output_path


def load_articles(site_name: str | None = None) -> list[dict]:
    """Load previously scraped articles from JSON files.

    Args:
        site_name: If provided, load only that site's articles.
                   If None, load all available article files.
    """
    articles = []
    if site_name:
        path = DATA_DIR / f"{site_name}_articles.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                articles = json.load(f)
    else:
        for path in DATA_DIR.glob("*_articles.json"):
            with open(path, encoding="utf-8") as f:
                articles.extend(json.load(f))
    return articles


def run_scraper(
    sites: list[str] | None = None,
    max_articles_per_site: int = 50,
) -> list[dict]:
    """Run the scraper for specified sites.

    Args:
        sites: List of site keys to scrape. Defaults to DEFAULT_SITES.
        max_articles_per_site: Maximum articles to scrape per site.

    Returns:
        List of all scraped article dicts.
    """
    sites = sites or DEFAULT_SITES
    all_articles = []

    for site_key in sites:
        config = ALL_SITES.get(site_key)
        if not config:
            logger.warning(f"Unknown site: {site_key}, skipping")
            continue

        articles = scrape_site(config, max_articles=max_articles_per_site)
        if articles:
            save_articles(articles, site_key)
            all_articles.extend(articles)

    logger.info(f"Total articles scraped: {len(all_articles)}")
    return all_articles


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scrape fashion editorial articles")
    parser.add_argument(
        "--sites",
        nargs="+",
        choices=list(ALL_SITES.keys()),
        default=DEFAULT_SITES,
        help="Sites to scrape",
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=50,
        help="Max articles per site",
    )
    args = parser.parse_args()

    run_scraper(sites=args.sites, max_articles_per_site=args.max_articles)
