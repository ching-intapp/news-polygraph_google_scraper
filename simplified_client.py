import httpx
from httpx import AsyncClient

class ScraperClient:
    def __init__(self, scraper_id, headers=None, timeout=30):
        self.scraper_id = scraper_id
        self.headers = headers if headers is not None else {}
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=self.timeout, headers=self.headers)

    async def get(self, url, params=None):
        response = await self.client.get(url, params=params)
        response.raise_for_status()
        return response

    async def post(self, url, data=None, json=None):
        response = await self.client.post(url, data=data, json=json)
        response.raise_for_status()
        return response

    async def close(self):
        await self.client.aclose()