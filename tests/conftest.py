import pytest
import asyncio

@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def sample_signal():
    return {
        "ticker": "SBER",
        "action": "buy",
        "quantity": 10,
        "price": 280.50,
        "confidence": 0.75
    }
