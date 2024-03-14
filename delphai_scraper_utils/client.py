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
from .metrics import request_response_received
from time import perf_counter


HTTP_RETRY_EXCEPTION_TYPES = (
    retry_if_exception_type(TimeoutException)
    | retry_if_exception_type(ConnectError)
    | retry_if_exception_type(ReadTimeout)
    | retry_if_exception_type(ProxyError)
    | retry_if_exception_type(RemoteProtocolError)
    | retry_if_exception_type(ssl.SSLZeroReturnError)
    | retry_if_exception_type(ssl.SSLError)
)


class AsyncRetryClient(AsyncClient):
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
        max_retry_attemps: int = 0,
        retry_wait_multiplier: int = 1,
        retry_wait_max: int = 1,
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

        # Wrap request in retry decorator
        self.request = retry(
            stop=stop_after_attempt(max_retry_attemps),
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
        if not self.persist_cookies:
            self._cookies = Cookies(None)

        response_time = -perf_counter()
        response = await super().request(
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
        response_time += perf_counter()
        if self.scraper_id:
            request_response_received(
                scraper_id=self.scraper_id,
                response_time=response_time,
            )
        return response
