from src.kg_qa.api import router
from src.kg_qa.schemas import QAResponse


def test_stream_route_and_total_duration_contract_exist():
    routes = {(route.path, tuple(sorted(route.methods or []))) for route in router.routes}
    assert ("/kg-qa/stream", ("POST",)) in routes
    response = QAResponse(run_id="qa-test", status="COMPLETED", total_duration_ms=12.5)
    assert response.total_duration_ms == 12.5

