"""
Site configurations for fashion editorial scraping.

Each source defines the site name, base URL, article listing pages,
and CSS selectors for extracting article content.

All sites listed here have been verified to allow scraping via their
robots.txt as of February 2026.
"""

from dataclasses import dataclass, field


@dataclass
class SiteConfig:
    """Configuration for scraping a single editorial site."""

    name: str
    base_url: str
    listing_urls: list[str]
    article_link_selector: str
    title_selector: str
    body_selector: str
    author_selector: str = ""
    date_selector: str = ""
    category_selector: str = ""
    # CSS selectors for elements to remove from body before extracting text
    remove_selectors: list[str] = field(default_factory=list)
    # CSS selectors for text-bearing elements within the body container.
    # When set, the scraper uses these instead of the default p/h2/h3/li extraction.
    content_selectors: list[str] = field(default_factory=list)
    # Delay between requests in seconds
    request_delay: float = 2.0


# ──────────────────────────────────────────────────────────────
# Men's fashion
# ──────────────────────────────────────────────────────────────

# Put This On (classic menswear) - robots.txt fully open
PUT_THIS_ON = SiteConfig(
    name="putthison",
    base_url="https://putthison.com",
    listing_urls=[
        "https://putthison.com/",
        "https://putthison.com/page/2/",
        "https://putthison.com/page/3/",
    ],
    article_link_selector=".entry-title a, h2 a, h4 a",
    title_selector="h1.entry-title, h1.wp-block-post-title, h1.post-title, h1",
    body_selector=".entry-content, .wp-block-post-content, .post-content",
    author_selector=".author, .byline",
    date_selector="time.entry-date, .post-date, time",
    remove_selectors=[".sharedaddy", ".jp-relatedposts", ".ad"],
    request_delay=2.0,
)

# The Fashionisto (outfit inspiration, styling tips) - robots.txt allows articles
THE_FASHIONISTO = SiteConfig(
    name="thefashionisto",
    base_url="https://www.thefashionisto.com",
    listing_urls=[
        "https://www.thefashionisto.com/category/style-guides/",
        "https://www.thefashionisto.com/category/mens-style/",
    ],
    article_link_selector="h2 a, h3 a, .entry-title a",
    title_selector="h1.entry-title, h1",
    body_selector=".entry-content",
    author_selector=".author-name, .byline",
    date_selector="time.entry-date, time",
    remove_selectors=[".ad", ".related-posts", ".wp-caption-text"],
    request_delay=2.5,
)

# Die, Workwear! (in-depth menswear writing) - robots.txt fully open
DIE_WORKWEAR = SiteConfig(
    name="dieworkwear",
    base_url="https://dieworkwear.com",
    listing_urls=[
        "https://dieworkwear.com/",
        "https://dieworkwear.com/page/2",
        "https://dieworkwear.com/page/3",
    ],
    article_link_selector=".post-title a, h2 a, h3 a, .entry-title a, article a",
    title_selector="h1.post-title, h1.entry-title, h1",
    body_selector=".post-body, .entry-content",
    author_selector=".author-name, .byline",
    date_selector="time, .post-date",
    remove_selectors=[".sharedaddy", ".related-posts"],
    request_delay=2.0,
)

# Permanent Style (classic tailoring and menswear) - robots.txt open for general crawlers
PERMANENT_STYLE = SiteConfig(
    name="permanentstyle",
    base_url="https://www.permanentstyle.com",
    listing_urls=[
        "https://www.permanentstyle.com/",
        "https://www.permanentstyle.com/page/2",
        "https://www.permanentstyle.com/page/3",
    ],
    article_link_selector="h3.entry-title a, .entry-title a, h2 a, h3 a",
    title_selector="h1.entry-title, h1",
    body_selector=".entry-content",
    author_selector=".author, .byline",
    date_selector="time.entry-date, .post-date, time",
    remove_selectors=[".sharedaddy", ".related-posts", ".ad"],
    request_delay=2.5,
)

# ──────────────────────────────────────────────────────────────
# Women's fashion
# ──────────────────────────────────────────────────────────────

# Corporette (professional women's workwear) - robots.txt nothing blocked
CORPORETTE = SiteConfig(
    name="corporette",
    base_url="https://corporette.com",
    listing_urls=[
        "https://corporette.com/",
        "https://corporette.com/page/2/",
        "https://corporette.com/page/3/",
    ],
    article_link_selector=".entry-title a, h2 a, h3 a",
    title_selector="h1.entry-title, h1",
    body_selector=".entry-content",
    author_selector=".author, .byline",
    date_selector="time.entry-date, time",
    remove_selectors=[".sharedaddy", ".related-posts", ".ad"],
    request_delay=2.0,
)

# Wardrobe Oxygen (capsule wardrobes, practical styling) - robots.txt fully open
WARDROBE_OXYGEN = SiteConfig(
    name="wardrobeoxygen",
    base_url="https://www.wardrobeoxygen.com",
    listing_urls=[
        "https://www.wardrobeoxygen.com/",
        "https://www.wardrobeoxygen.com/page/2/",
        "https://www.wardrobeoxygen.com/page/3/",
    ],
    article_link_selector=".entry-title a, h2 a, h3 a",
    title_selector="h1.entry-title, h1",
    body_selector=".entry-content",
    author_selector=".author-name, .byline",
    date_selector="time.entry-date, time",
    remove_selectors=[".ad", ".related-posts", ".sharedaddy"],
    request_delay=2.0,
)

# The Fashion Spot (fashion editorial, trends) - robots.txt fully open
THE_FASHION_SPOT = SiteConfig(
    name="thefashionspot",
    base_url="https://www.thefashionspot.com",
    listing_urls=[
        "https://www.thefashionspot.com/style-trends/",
        "https://www.thefashionspot.com/runway-news/",
    ],
    article_link_selector="a[href*='/style-trends/'], a[href*='/runway-news/'], h2 a, h3 a",
    title_selector="h1",
    body_selector=".entry-content, .article-body, .post-content",
    author_selector=".byline, .author-name",
    date_selector="time",
    remove_selectors=[".ad", ".related-posts", ".newsletter"],
    request_delay=2.5,
)

# ──────────────────────────────────────────────────────────────
# Unisex / gender-neutral
# ──────────────────────────────────────────────────────────────

# Coveteur (fashion culture, closet tours, trends) - robots.txt fully open
COVETEUR = SiteConfig(
    name="coveteur",
    base_url="https://www.coveteur.com",
    listing_urls=[
        "https://www.coveteur.com/fashion",
    ],
    article_link_selector=".widget__headline a, h2 a, h3 a",
    title_selector="h1.widget__headline, h1",
    body_selector=".body-description, .article-content, .entry-content",
    author_selector=".post-author, .social-author__name, .byline",
    date_selector=".social-date, .post-date, time",
    category_selector=".widget__section",
    remove_selectors=[".ad", ".related-posts", ".newsletter", ".media-photo-credit"],
    request_delay=2.5,
)

# Hypebeast (streetwear, sneakers, culture) - robots.txt allows articles
HYPEBEAST = SiteConfig(
    name="hypebeast",
    base_url="https://hypebeast.com",
    listing_urls=[
        "https://hypebeast.com/fashion",
        "https://hypebeast.com/style",
    ],
    article_link_selector="a[href*='/20']",
    title_selector="h1",
    body_selector=".post-body, .post-body-component, article .content",
    author_selector=".author-name, .byline",
    date_selector="time",
    remove_selectors=[".ad", ".related-posts", ".newsletter"],
    request_delay=3.0,
)

# i-D Magazine (cultural fashion, emerging talent) - robots.txt fully open
I_D_MAGAZINE = SiteConfig(
    name="i_d",
    base_url="https://i-d.co",
    listing_urls=[
        "https://i-d.co/",
        "https://i-d.co/category/fashion/",
    ],
    article_link_selector="a[href*='/article/']",
    title_selector="h1, .wp-block-post-title",
    body_selector=".wp-block-post-content, .entry-content",
    author_selector=".byline, .author-name",
    date_selector="time",
    remove_selectors=[".ad", ".related-posts", ".wp-block-id-magazine-image-slider-marquee"],
    request_delay=2.5,
)

# ──────────────────────────────────────────────────────────────
# Trends / Y2K
# ──────────────────────────────────────────────────────────────

# Refinery29 (trend reports, Y2K fashion) - robots.txt allows general crawlers, 1s crawl delay
REFINERY29 = SiteConfig(
    name="refinery29",
    base_url="https://www.refinery29.com",
    listing_urls=[
        # Category pages
        "https://www.refinery29.com/en-us/fashion",
        "https://www.refinery29.com/en-us/fashion?page=2",
        "https://www.refinery29.com/en-us/fashion?page=3",
        "https://www.refinery29.com/en-us/trends",
        "https://www.refinery29.com/en-us/trends?page=2",
        "https://www.refinery29.com/en-us/trends?page=3",
        # Targeted search results for Y2K and styling content
        "https://www.refinery29.com/en-us/search?q=y2k+fashion+style",
        "https://www.refinery29.com/en-us/search?q=nostalgic+fashion+trend",
        "https://www.refinery29.com/en-us/search?q=how+to+style+outfit",
    ],
    article_link_selector=".card a",
    title_selector="h1.title, h1",
    body_selector="section.body",
    author_selector=".byline.main-contributors, .featured-byline",
    date_selector=".byline.modified",
    remove_selectors=[".ad", ".related-posts", ".newsletter-module", ".inline-promo"],
    content_selectors=[".section-text"],
    request_delay=2.0,
)

# All available site configs
ALL_SITES: dict[str, SiteConfig] = {
    # Men's
    "putthison": PUT_THIS_ON,
    "thefashionisto": THE_FASHIONISTO,
    "dieworkwear": DIE_WORKWEAR,
    "permanentstyle": PERMANENT_STYLE,
    # Women's
    "corporette": CORPORETTE,
    "wardrobeoxygen": WARDROBE_OXYGEN,
    "thefashionspot": THE_FASHION_SPOT,
    # Unisex
    "coveteur": COVETEUR,
    "hypebeast": HYPEBEAST,
    "i_d": I_D_MAGAZINE,
    # Trends / Y2K
    "refinery29": REFINERY29,
}

# Default sites to scrape (mix of men's, women's, and unisex)
DEFAULT_SITES = [
    "putthison",
    "thefashionisto",
    "dieworkwear",
    "permanentstyle",
    "corporette",
    "wardrobeoxygen",
    "thefashionspot",
    "coveteur",
    "hypebeast",
    "i_d",
    "refinery29",
]
