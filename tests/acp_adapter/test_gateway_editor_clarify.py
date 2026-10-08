"""A canonical clarify question reaches the ACP editor and its answer reaches the model.

Real gateway daemon, real ACP stdio process, loopback model that calls the real clarify tool. The
session is created by the CLI (whose frozen tools include clarify), so the turn blocks on a
question the editor must be able to answer: live while its own prompt runs, or restored from the
snapshot when it attaches after another surface's turn already asked.
"""
import asyncio
import json

import pytest

from tests.acp_adapter.test_gateway_sessions import editor, viewer


@pytest.mark.platforms("linux")
@pytest.mark.asyncio
@pytest.mark.parametrize("form, restored", [(False, False), (True, True)],
                         ids=["live-permission-card", "restored-form-elicitation"])
async def test_editor_answers_canonical_clarify_through_clarify_respond(
        daemon, tmp_path, model_peer, form, restored):
    from tests.gateway.fixtures.authority_clarify_peer import ModelPeer as ClarifyPeer

    model_peer.RequestHandlerClass = ClarifyPeer
    async with viewer(daemon) as ws:
        created = await ws.rpc("session.create", request_id="acp-clarify", source="cli", cwd=str(tmp_path))
        sid = created["session_id"]
        turn = None
        if restored:
            accepted = await ws.rpc("prompt.submit", session_id=sid, input_id="cli-turn", text="Ask my color")
            async with asyncio.timeout(30):
                while not (await ws.rpc("session.resume", session_id=sid))["prompts"]:
                    await asyncio.sleep(.05)
        async with editor(daemon, tmp_path) as acp:
            capabilities = {"elicitation": {"form": {}}} if form else {}
            assert "result" in await acp.rpc("initialize", protocolVersion=1, clientCapabilities=capabilities)
            assert "result" in await acp.rpc("session/load", cwd=str(tmp_path), sessionId=sid, mcpServers=[])
            if not restored:
                turn = asyncio.create_task(acp.rpc("session/prompt", sessionId=sid,
                                                   prompt=[{"type": "text", "text": "Ask my color"}]))
            method = "elicitation/create" if form else "session/request_permission"
            async with asyncio.timeout(30):
                while not any(f.get("method") == method for f in acp.frames):
                    assert turn is None or not turn.done(), turn.result()
                    await asyncio.sleep(.05)
            request = next(f for f in acp.frames if f.get("method") == method)
            params = request["params"]
            assert params["sessionId"] == sid
            if form:
                assert params["message"].startswith("Pick a color")
                assert params["requestedSchema"]["properties"]["answer"]["enum"][1] == "green"
                result = {"action": "accept", "content": {"answer": "green"}}
            else:
                assert params["toolCall"]["title"].startswith("Pick a color")
                option = next(o for o in params["options"] if o["name"] == "green")
                result = {"outcome": {"outcome": "selected", "optionId": option["optionId"]}}
            acp.process.stdin.write(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}).encode()
                                    + b"\n")
            await acp.process.stdin.drain()
            if turn is not None:
                reply = await asyncio.wait_for(turn, 40)
                assert reply.get("result", {}).get("stopReason") == "end_turn", reply
            else:
                async with asyncio.timeout(40):
                    while (await ws.rpc("prompt.receipt", session_id=sid,
                                        admission_id=accepted["admission_id"]))["status"] != "terminal":
                        await asyncio.sleep(.05)
    tool_results = [m for messages in model_peer.requests for m in messages if m["role"] == "tool"]
    assert any("green" in json.dumps(m) for m in tool_results), tool_results
