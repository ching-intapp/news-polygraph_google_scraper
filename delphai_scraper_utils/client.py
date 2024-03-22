from httpx import (
    TimeoutException,
    ConnectError,
    ReadTimeout,
    ProxyError,
    RemoteProtocolError,
    AsyncClient,
)
from httpx._client import USE_CLIENT_DEFAULT, UseClientDefault
from httpx._config import (
    DEFAULT_LIMITS,
    DEFAULT_MAX_REDIRECTS,
    DEFAULT_TIMEOUT_CONFIG,
    Limits,
)
from httpx._models import Response, Cookies
from httpx._transports.base import AsyncBaseTransport
from httpx._types import (
    AuthTypes,
    CertTypes,
    CookieTypes,
    HeaderTypes,
    ProxiesTypes,
    QueryParamTypes,
    RequestContent,
    RequestData,
    RequestFiles,
    TimeoutTypes,
    URLTypes,
    VerifyTypes,
)
from tenacity import (
    retry,
    wait_random_exponential,
    stop_after_attempt,
    retry_if_exception_type,
)
import ssl
from typing import Callable, Any, List, Mapping, Union
from .metrics import request_timer
from aiocache import cached
from urllib.robotparser import RobotFileParser
from urllib.parse import urlparse, urljoin


HTTP_RETRY_EXCEPTION_TYPES = (
    retry_if_exception_type(TimeoutException)
    | retry_if_exception_type(ConnectError)
    | retry_if_exception_type(ReadTimeout)
    | retry_if_exception_type(ProxyError)
    | retry_if_exception_type(RemoteProtocolError)
    | retry_if_exception_type(ssl.SSLZeroReturnError)
    | retry_if_exception_type(ssl.SSLError)
)


class ScraperClient(AsyncClient):
    def __init__(
        self,
        *,
        auth: AuthTypes = None,
        params: QueryParamTypes = None,
        headers: HeaderTypes = None,
        cookies: CookieTypes = None,
        verify: VerifyTypes = True,
        cert: CertTypes = None,
        http1: bool = True,
        http2: bool = False,
        proxies: ProxiesTypes = None,
        mounts: Mapping[str, AsyncBaseTransport] = None,
        timeout: TimeoutTypes = DEFAULT_TIMEOUT_CONFIG,
        follow_redirects: bool = False,
        limits: Limits = DEFAULT_LIMITS,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
        event_hooks: Mapping[str, List[Callable]] = None,
        base_url: URLTypes = "",
        transport: AsyncBaseTransport = None,
        app: Callable = None,
        trust_env: bool = True,
        persist_cookies: bool = False,
        scraper_id: Union[str, None] = None,
        max_retry_attempts: int = 3,
        retry_wait_multiplier: int = 1,
        retry_wait_max: int = 1,
        ignore_robots_txt: bool = False,
    ):
        super().__init__(
            auth=auth,
            params=params,
            headers=headers,
            cookies=cookies,
            verify=verify,
            cert=cert,
            http1=http1,
            http2=http2,
            proxies=proxies,
            mounts=mounts,
            timeout=timeout,
            follow_redirects=follow_redirects,
            limits=limits,
            max_redirects=max_redirects,
            event_hooks=event_hooks,
            base_url=base_url,
            transport=transport,
            app=app,
            trust_env=trust_env,
        )
        self.scraper_id = scraper_id
        self.persist_cookies = persist_cookies
        self.ignore_robots_txt = ignore_robots_txt

        # Wrap request in retry decorator
        self.request = retry(
            stop=stop_after_attempt(max_retry_attempts),
            wait=wait_random_exponential(
                multiplier=retry_wait_multiplier, max=retry_wait_max
            ),
            retry=HTTP_RETRY_EXCEPTION_TYPES,
            reraise=True,
        )(self.request)

    async def request(
        self,
        method: str,
        url: URLTypes,
        *,
        content: RequestContent = None,
        data: RequestData = None,
        files: RequestFiles = None,
        json: Any = None,
        params: QueryParamTypes = None,
        headers: HeaderTypes = None,
        cookies: CookieTypes = None,
        auth: Union[AuthTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        timeout: Union[TimeoutTypes, UseClientDefault] = USE_CLIENT_DEFAULT,
        extensions: dict = None,
    ) -> Response:
        if not self.ignore_robots_txt:
            user_agent = (headers or {}).get("user-agent", "*")
            if not await self.is_allowed_by_robots_text(url, user_agent):
                return Response(403)

        if not self.persist_cookies:
            self._cookies = Cookies(None)
        with request_timer(scraper_id=self.scraper_id):
            return await super().request(
                method,
                url,
                content=content,
                data=data,
                files=files,
                json=json,
                params=params,
                headers=headers,
                cookies=cookies,
                auth=auth,
                follow_redirects=follow_redirects,
                timeout=timeout,
                extensions=extensions,
            )

    @cached()
    async def get_robots_text_parser(
        self, base_url: str, user_agent: str
    ) -> RobotFileParser:
        robots_text_url = urljoin(base_url, "robots.txt")
        robot_file_parser = RobotFileParser(robots_text_url)
        with request_timer(scraper_id=self.scraper_id):
            response = await super().request(
                "GET",
                robots_text_url,
                headers={"user-agent": user_agent},
            )

        if response.status_code == 404:
            # If no robots.txt is found, consider everything allowed
            robot_file_parser.allow_all = True
        else:
            response.raise_for_status()
            robot_file_parser.parse(response.iter_lines())

        return robot_file_parser

    async def is_allowed_by_robots_text(self, url: str, user_agent: str) -> bool:
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        robots_text_parser = await self.get_robots_text_parser(base_url, user_agent)

        return robots_text_parser.can_fetch(url=url, useragent=user_agent)
