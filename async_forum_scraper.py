import aiohttp
import asyncio
import json
import re
import time
import random
from bs4 import BeautifulSoup
from typing import Dict, List, Callable, Any, Optional, Tuple, Set

class AsyncForumScraper:
    """Asynchronous scraper for Toribash forum posts."""
    
    def __init__(self, base_url: str = "https://forum.toribash.com", 
                 progress_callback: Optional[Callable] = None):
        """Initialize the scraper with base URL and optional progress callback."""
        self.base_url = base_url
        self.progress_callback = progress_callback
        self.session = None
        
        # Regex patterns for cleaning content
        self.signature_patterns = [
            r'\[SIGPIC\].*?\[\/SIGPIC\]',
            r'__________________[\s\S]*?(?=\[|\'|\n\n|$)',
            r'<!--.*?-->',
            r'\[attachment=.*?\]',
            r'\[quote.*?\].*?\[\/quote\]',
            r'\[img\].*?\[\/img\]',
            r'\[url=.*?\].*?\[\/url\]',
            r'\[\/?[a-z]+\]'
        ]
        
        # Patterns for dates and edited lines
        self.date_patterns = [
            r'\d{2}-\d{2}-\d{4}, \d{2}:\d{2} [AP]M',
            r'\d{2}/\d{2}/\d{4} \d{2}:\d{2} [AP]M',
            r'\d{2}/\d{2}/\d{4}'
        ]
        
        self.edited_patterns = [
            r'Last edited by .*? at .*?'
        ]
        
        # Patterns for attachments
        self.attachment_patterns = [
            r'Attached Files.*?(?=\n\n|$)',
            r'Attached Thumbnails.*?(?=\n\n|$)'
        ]
    
    async def __aenter__(self):
        """Create aiohttp session when entering context manager."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close aiohttp session when exiting context manager."""
        if self.session:
            await self.session.close()
            self.session = None
    
    def _report_progress(self, progress_type: str, data: Dict[str, Any]):
        """Report progress via callback if provided."""
        if self.progress_callback:
            self.progress_callback(progress_type, data)
    
    async def _fetch_post(self, post_id: int, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
        """Fetch a single post by ID with retry logic and rate limiting."""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        # Use showpost.php for direct post access
        url = f"{self.base_url}/showpost.php?p={post_id}"
        max_retries = 3
        retry_delay = 1.0
        
        async with semaphore:
            # Add random delay for rate limiting
            await asyncio.sleep(random.uniform(0.1, 0.5))
            
            for attempt in range(max_retries):
                start_time = time.time()
                try:
                    async with self.session.get(url, timeout=10) as response:
                        response_time = time.time() - start_time
                        
                        # Report connection status
                        self._report_progress("connection", {
                            "post_id": post_id,
                            "success": response.status == 200,
                            "status": response.status,
                            "response_time": response_time,
                            "url": url
                        })
                        
                        if response.status == 200:
                            html = await response.text()
                            
                            # Process HTML in memory without saving to disk
                            
                            result = await self._extract_post_data(html, post_id, url)
                            
                            if result:
                                self._report_progress("post_processed", {
                                    "post_id": post_id,
                                    "success": True,
                                    **result
                                })
                                return result
                            else:
                                self._report_progress("post_processed", {
                                    "post_id": post_id,
                                    "success": False,
                                    "error": "No post data found"
                                })
                                return {"post_id": post_id, "error": "No post data found"}
                        else:
                            error_msg = f"HTTP Error: {response.status}"
                            if attempt < max_retries - 1:
                                # Exponential backoff
                                retry_delay *= 2
                                await asyncio.sleep(retry_delay)
                            else:
                                self._report_progress("post_processed", {
                                    "post_id": post_id,
                                    "success": False,
                                    "error": error_msg
                                })
                                return {"post_id": post_id, "error": error_msg}
                
                except asyncio.TimeoutError:
                    error_msg = "Timeout Error"
                    if attempt < max_retries - 1:
                        retry_delay *= 2
                        await asyncio.sleep(retry_delay)
                    else:
                        self._report_progress("post_processed", {
                            "post_id": post_id,
                            "success": False,
                            "error": error_msg
                        })
                        return {"post_id": post_id, "error": error_msg}
                
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    if attempt < max_retries - 1:
                        retry_delay *= 2
                        await asyncio.sleep(retry_delay)
                    else:
                        self._report_progress("post_processed", {
                            "post_id": post_id,
                            "success": False,
                            "error": error_msg
                        })
                        return {"post_id": post_id, "error": error_msg}
    
    async def _extract_post_data(self, html: str, post_id: int, url: str) -> Optional[Dict[str, Any]]:
        """Extract post data from HTML content."""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Check if the page contains login requirement message
        if "You are not logged in or you do not have permission to access this page" in html:
            # This is not an error - we can still extract data from the page
            print(f"Note: Post {post_id} shows login message but we can still extract data")
        
        # Find post container - updated to match the actual HTML structure
        post_container = soup.find('div', id=f'post_message_{post_id}')
        if not post_container:
            # Try alternative structure - this is the correct one based on the HTML
            post_container = soup.find('div', class_='showthread-post', id=f'post_message_{post_id}')
            if not post_container:
                # Try just the class without the ID
                post_container = soup.find('div', class_='showthread-post')
        
        if not post_container:
            # Debug info
            print(f"Could not find post container for post {post_id}")
            return None
        
        # Extract thread ID from URL
        thread_id_match = re.search(r'p=(\d+)', url)
        thread_id = thread_id_match.group(1) if thread_id_match else None
        
        # Extract username - updated selector
        username_elem = soup.find('a', class_='postbit-username')
        username = username_elem.text.strip() if username_elem else "Unknown"
        
        # Extract date - updated selector
        date_elem = soup.find('div', class_='showthread-postdateold')
        date = date_elem.text.strip() if date_elem else ""
        
        # Extract content - updated selector
        # The content is directly in the post_container
        raw_content = post_container.text.strip() if post_container else ""
        
        # Clean content
        content = self._clean_content(raw_content)
        
        # Skip if content is too short, but be more lenient
        if len(content) < 10 or len(content.split()) < 3:
            return None
        
        # Extract thread title
        title_elem = soup.find('title')
        title = title_elem.text.strip() if title_elem else ""
        
        # Extract thread ID from thread link if not found in URL
        if not thread_id:
            thread_link = soup.find('a', string=lambda s: s and 'Thread' in s)
            if thread_link and 'href' in thread_link.attrs:
                thread_href = thread_link['href']
                thread_id_match = re.search(r't=(\d+)', thread_href)
                if not thread_id_match:
                    thread_id_match = re.search(r'p=(\d+)', thread_href)
                thread_id = thread_id_match.group(1) if thread_id_match else None
        
        return {
            "post_id": str(post_id),
            "thread_id": thread_id,
            "username": username,
            "date": date,
            "content": content,
            "url": url,
            "title": title
        }
    
    def _clean_content(self, content: str) -> str:
        """Clean post content by removing signatures, formatting, etc."""
        # Apply signature patterns
        for pattern in self.signature_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove dates
        for pattern in self.date_patterns:
            content = re.sub(pattern, '', content)
        
        # Remove edited lines
        for pattern in self.edited_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)
        
        # Remove attachments
        for pattern in self.attachment_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE | re.DOTALL)
        
        # Clean up whitespace
        content = re.sub(r'\s+', ' ', content).strip()
        content = re.sub(r'\n\s*\n', '\n\n', content)
        
        return content
    
    async def scrape_range(self, start_id: int, end_id: int, max_connections: int = 10) -> Dict[str, Any]:
        """Scrape a range of post IDs with controlled concurrency."""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        # Create semaphore to limit concurrent connections
        semaphore = asyncio.Semaphore(max_connections)
        
        # Report start of scraping
        self._report_progress("start", {
            "total_posts": end_id - start_id + 1,
            "start_id": start_id,
            "end_id": end_id,
            "max_connections": max_connections
        })
        
        # Create tasks for each post ID
        tasks = [self._fetch_post(post_id, semaphore) for post_id in range(start_id, end_id + 1)]
        results = await asyncio.gather(*tasks)
        
        # Filter successful results
        posts = [post for post in results if "error" not in post]
        failed_ids = [int(post["post_id"]) for post in results if "error" in post]
        
        # Report completion
        self._report_progress("complete", {
            "total_posts": end_id - start_id + 1,
            "successful_posts": len(posts),
            "failed_posts": len(failed_ids)
        })
        
        return {
            "posts": posts,
            "failed_ids": failed_ids
        }
    
    def save_data(self, posts: List[Dict[str, Any]], failed_ids: List[int], filename: str):
        """Save scraped data to a JSON file."""
        data = {
            "posts": posts,
            "failed_ids": failed_ids,
            "metadata": {
                "total_posts": len(posts),
                "total_failed": len(failed_ids),
                "timestamp": time.time()
            }
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return filename

# Example usage
async def main():
    async with AsyncForumScraper() as scraper:
        results = await scraper.scrape_range(10000, 10100, 10)
        scraper.save_data(results['posts'], results['failed_ids'], 'forum_data.json')

if __name__ == "__main__":
    asyncio.run(main())