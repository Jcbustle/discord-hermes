"""Canonical response retry identity is fixed before concurrent admission."""
import asyncio

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


@pytest.mark.asyncio
async def test_concurrent_changed_conversation_cannot_replace_response_key(api, owner):
    api._api_key = 'identity-fixture'
    started = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def handle(event):
        from gateway.session_results import execution_result
        calls.append(event.text)
        started.set()
        if event.text == 'first':
            await release.wait()
        execution_result.get()['result'] = {'final_response': 'first answer', 'messages': []}
        return 'first answer'

    owner.runner._handle_message = handle
    app = web.Application()
    app.router.add_post('/v1/responses', api._handle_responses)
    headers = {'Idempotency-Key': 'same-request', 'X-Hermes-Session-Key': 'first-room',
               'Authorization': 'Bearer identity-fixture'}
    async with TestClient(TestServer(app)) as client:
        first = asyncio.create_task(client.post('/v1/responses', json={'input': 'first'}, headers=headers))
        await asyncio.wait_for(started.wait(), 5)
        try:
            changed = await client.post('/v1/responses', json={'input': 'different'},
                headers={**headers, 'X-Hermes-Session-Key': 'second-room'})
            assert changed.status == 409
        finally:
            release.set()
        original = await first
        assert original.status == 200
        saved = await original.json()
        replay = await client.post('/v1/responses', json={'input': 'first'}, headers=headers)
        assert await replay.json() == saved
        changed_target = await client.post('/v1/responses', json={'input': 'first'},
            headers={**headers, 'X-Hermes-Session-Key': 'second-room'})
        assert changed_target.status == 409
    assert calls == ['first']
