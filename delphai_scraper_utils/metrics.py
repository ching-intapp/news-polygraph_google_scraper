from prometheus_client import Counter, Histogram
from contextlib import contextmanager
from time import perf_counter
from typing import Optional

scraper_requests_per_company = Histogram(
    name="scraper_requests_per_company",
    documentation="Number of requests sent by a scraper per run",
    labelnames=["scraper_id"],
)

scraper_requests_response_time = Histogram(
    name="scraper_requests_response_time",
    documentation="Time requests to scraper sources took",
    labelnames=["scraper_id"],
)

scraper_requests_count = Counter(
    name="scraper_requests_count",
    documentation="Total count of requests per source",
    labelnames=["scraper_id"],
)


def request_response_received(*, scraper_id: str, response_time: float):
    labels = dict(scraper_id=scraper_id)

    scraper_requests_count.labels(**labels).inc()
    scraper_requests_response_time.labels(**labels).observe(response_time)


@contextmanager
def request_timer(scraper_id: Optional[str] = None):
    try:
        response_time = -perf_counter()
        yield
    finally:
        response_time += perf_counter()
    if scraper_id:
        request_response_received(
            scraper_id=scraper_id,
            response_time=response_time,
        )


scraper_datapoints = Histogram(
    name="scraper_datapoints",
    documentation="Number of data points found by a scraper per run",
    labelnames=["scraper_id", "data_type"],
)


def datapoints_found(*, data_type: str, amount: int, scraper_id: str = None):
    labels = dict(scraper_id=scraper_id, data_type=data_type)
    scraper_datapoints.labels(**labels).observe(amount)
