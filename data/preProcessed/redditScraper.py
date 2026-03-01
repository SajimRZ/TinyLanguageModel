"""
Reddit Scraper - Improved Version
- Smarter rate limit handling
- Writes after every post
- Retry with exponential backoff
"""

import requests
import time
import random
from datetime import datetime
import re


class RedditScraper:
    def __init__(self, subreddit, max_posts=100, max_comments_per_post=50,
                 include_posts=True, output_file=None,
                 max_retries=5):

        self.subreddit = subreddit
        self.max_posts = max_posts
        self.max_comments_per_post = max_comments_per_post
        self.include_posts = include_posts
        self.output_file = output_file or f"reddit_{subreddit}.txt"
        self.max_retries = max_retries

        self.session = requests.Session()
        self.session.headers.update({
            # More realistic + unique user agent
            'User-Agent': f'RedditScraperBot/1.0 (by u/Introsaico)'
        })

    # =============================
    # Public Method
    # =============================
    def scrape(self):
        print(f"🔍 Scraping r/{self.subreddit}")
        print(f"Posts: {self.max_posts}")
        print(f"Comments per post: {self.max_comments_per_post}")
        print("-" * 40)

        posts_scraped = 0
        after = None

        # Clear file at start
        # open(self.output_file, "w", encoding="utf-8").close()

        while posts_scraped < self.max_posts:
            posts = self._get_posts(after=after,
                                    limit=min(100, self.max_posts - posts_scraped))

            if not posts:
                print("⚠️ No more posts available")
                break

            for post in posts:
                post_data = self._parse_post(post)

                # Write immediately after each post
                if self.include_posts == True:
                    self._write_item(post_data)

                if self.max_comments_per_post > 0:
                    comments = self._get_comments(
                        post['data']['id'],
                        self.max_comments_per_post
                    )
                    for comment in comments:
                        self._write_item(comment)

                posts_scraped += 1
                print(f"✓ {posts_scraped}/{self.max_posts} - {post_data['title'][:60]}")

                if posts_scraped >= self.max_posts:
                    break

                # Random delay (human-like)
                time.sleep(random.uniform(1.5, 4.0))

            after = posts[-1]['data']['name'] if posts else None

            # Longer delay between pages
            time.sleep(random.uniform(5, 12))

        print(f"\n✅ Finished scraping {posts_scraped} posts")
        print(f"Saved to: {self.output_file}")

    # =============================
    # Request with Retry + Backoff
    # =============================
    def _request_with_retry(self, url, params):
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, params=params, timeout=15)

                # Handle rate limiting
                if response.status_code == 429:
                    wait = (2 ** attempt) + random.uniform(1, 3)
                    print(f"⚠️ Rate limited. Sleeping {wait:.1f}s...")
                    time.sleep(wait)
                    continue

                # Retry on server errors
                if response.status_code >= 500:
                    wait = (2 ** attempt)
                    print(f"⚠️ Server error {response.status_code}. Retrying in {wait}s")
                    time.sleep(wait)
                    continue

                response.raise_for_status()
                return response.json()

            except requests.RequestException as e:
                wait = (2 ** attempt)
                print(f"⚠️ Request failed ({e}). Retrying in {wait}s")
                time.sleep(wait)

        print("❌ Max retries exceeded.")
        return None

    # =============================
    # Fetch Posts
    # =============================
    def _get_posts(self, after=None, limit=100):
        url = f"https://old.reddit.com/r/{self.subreddit}/top.json"
        params = {
            'limit': limit,
            't': "all",
            'raw_json': 1
        }

        if after:
            params['after'] = after

        data = self._request_with_retry(url, params)
        if not data:
            return []

        return data.get('data', {}).get('children', [])

    # =============================
    # Fetch Comments
    # =============================
    def _get_comments(self, post_id, max_comments):
        url = f"https://old.reddit.com/r/{self.subreddit}/comments/{post_id}.json"
        params = {'raw_json': 1, 'limit': max_comments}

        data = self._request_with_retry(url, params)
        if not data or len(data) < 2:
            return []

        comments = []
        for comment in data[1]['data']['children'][:max_comments]:
            if comment['kind'] == 't1':
                parsed = self._parse_comment(comment)
                if parsed:
                    comments.append(parsed)

        return comments

    # =============================
    # Parsing
    # =============================
    def _parse_post(self, post):
        data = post['data']
        return {
            'type': 'POST',
            'title': self._clean_text(data.get('title', '')),
            'text': self._clean_text(data.get('selftext', '')),
            'score': data.get('score', 0),
            'author': data.get('author', '[deleted]'),
            'created': datetime.fromtimestamp(
                data.get('created_utc', 0)
            ).strftime('%Y-%m-%d')
        }

    def _parse_comment(self, comment):
        data = comment['data']
        body = data.get('body', '')

        if body in ['[deleted]', '[removed]', '']:
            return None

        return {
            'type': 'COMMENT',
            'text': self._clean_text(body),
            'score': data.get('score', 0),
            'author': data.get('author', '[deleted]'),
            'created': datetime.fromtimestamp(
                data.get('created_utc', 0)
            ).strftime('%Y-%m-%d')
        }

    # =============================
    # Text Cleaning
    # =============================
    def _clean_text(self, text):
        if not text:
            return ""
        # Remove markdown links but keep link text
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        # Remove raw URLs (http, https, www)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Remove Reddit usernames (u/username or /u/username)
        text = re.sub(r'/?u/[A-Za-z0-9_-]+', '', text)
        # Remove subreddit mentions (optional)
        text = re.sub(r'/?r/[A-Za-z0-9_-]+', '', text)
        # Collapse multiple newlines into ONE newline
        text = re.sub(r'\n+', '\n', text)
        # Collapse multiple spaces
        text = re.sub(r'[ \t]{2,}', ' ', text)
        # Strip leading/trailing whitespace
        return text.strip()

    # =============================
    # Write Immediately (Streaming)
    # =============================
    def _write_item(self, item):
        with open(self.output_file, 'a', encoding='utf-8') as f:
            if item['type'] == 'POST':
                f.write(f"{item['title']}\n")
                if item['text']:
                    f.write(f"{item['text']}\n")
                f.write("\n")

            elif item['type'] == 'COMMENT':
                if len(item['text']) > 50:
                    f.write(f"{item['text']}\n")


# Example usage
if __name__ == "__main__":
    scraper = RedditScraper(
        subreddit="cursedcomments",
        max_posts=10,
        max_comments_per_post=100,
        include_posts=False,
        output_file="redditScrapped.txt"
    )

    scraper.scrape()
