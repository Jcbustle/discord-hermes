"""Structured results are claim-owned and visible only after settlement."""
import pytest

from hermes_state import SessionDB
from hermes_state_runtime import (
    RuntimeStoreError, admit_session_input, begin_runtime_epoch,
    claim_session_input, recover_session_inputs,
)


def test_result_survives_restart_without_reexecuting(tmp_path):
    from gateway.session_results import retain_result, admission_result
    path = tmp_path / 'state.db'
    db = SessionDB(path)
    db.create_session('api-session', source='api_server')
    epoch = begin_runtime_epoch(db, instance_id='first')
    admitted = admit_session_input(db, epoch=epoch, principal_id='api', session_id='api-session',
                                   request_id='retry', payload={'text': 'hello'})
    row = claim_session_input(db, epoch=epoch, session_id='api-session')
    result = {'final_response': 'reply', 'messages': [], 'usage': {'input_tokens': 7, 'output_tokens': 3}}
    retain_result(db, epoch=epoch, row=row, result=result)
    assert admission_result(db, admitted['admission_id']) == result
    db.close()
    db = SessionDB(path)
    try:
        epoch = begin_runtime_epoch(db, instance_id='second')
        recover_session_inputs(db, epoch=epoch)
        retried = admit_session_input(db, epoch=epoch, principal_id='api', session_id='api-session',
                                      request_id='retry', payload={'text': 'hello'})
        assert retried['admission_id'] == admitted['admission_id']
        assert admission_result(db, retried['admission_id']) == result
        assert claim_session_input(db, epoch=epoch, session_id='api-session') is None
    finally:
        db.close()


def test_stale_result_cannot_overwrite_settled_or_unknown_claim(tmp_path):
    from gateway.session_results import retain_result, admission_result
    db = SessionDB(tmp_path / 'state.db')
    try:
        db.create_session('api-session', source='api_server')
        epoch = begin_runtime_epoch(db, instance_id='first')
        admit_session_input(db, epoch=epoch, principal_id='api', session_id='api-session',
                            request_id='one', payload={'text': 'hello'})
        row = claim_session_input(db, epoch=epoch, session_id='api-session')
        retain_result(db, epoch=epoch, row=row, result={'final_response': 'first'})
        with pytest.raises(RuntimeStoreError, match='stale_generation'):
            retain_result(db, epoch=epoch, row=row, result={'final_response': 'late'})
        assert admission_result(db, row['admission_id']) == {'final_response': 'first'}
        newer = begin_runtime_epoch(db, instance_id='second')
        with pytest.raises(RuntimeStoreError, match='stale_epoch'):
            retain_result(db, epoch=epoch, row=row, result={})
        assert newer != epoch
    finally:
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('fails', [True, False], ids=['agent-raised', 'plain-reply'])
async def test_a_turn_that_failed_before_running_settles_failed_on_every_surface(tmp_path, monkeypatch, fails):
    """A non-API admission whose agent turn raised before producing a TurnRunner result (agent
    initialization failure) must not settle ``completed`` with the apology as its output: ACP would
    answer the editor ``end_turn`` and a finite CLI run would exit 0. A handler reply that is not a
    failure (a notice, a command answer) still completes."""
    import asyncio
    from types import SimpleNamespace
    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_contract import Principal, SessionRef, Submission
    from hermes_state_runtime import get_session_admission

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    db = SessionDB(tmp_path / 'state.db')
    db.create_session('s', source='telegram')
    gateway = object.__new__(GatewayRunner)

    async def stop_typing(event, source):
        return None
    gateway._hmwa_stop_typing_for_turn = stop_typing
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='c', user_id='human')

    async def handle(event):
        if not fails:
            return 'Noted.'
        # The production except-body of the agent turn (agent construction raised).
        return await gateway._hmwa_agent_error_reply(RuntimeError('provider init failed'), event, source, None,
                                                     'k', gateway._PreparedTurn([], '', None, None, None, None))
    runner = SimpleNamespace(_draining=False, config=SimpleNamespace(multiplex_profiles=False),
                             _adapter_for_source=lambda source: None, _handle_message=handle)
    authority = SessionAuthority(runner, profile_id='p', instance_id='owner', db=db,
                                 epoch=begin_runtime_epoch(db, instance_id='owner'))
    authority.sessions['s'] = LiveSession(source, 'route')
    frames = []
    authority.sessions['s'].event_stream.observers.add(frames.append)
    try:
        receipt = await authority.submit(Principal('human', 'p', frozenset({'session:submit'}), 't'),
                                         Submission('r1', SessionRef('p', 's'), {'text': 'hello'}, 'queue'))
        await asyncio.wait_for(authority.sessions['s'].task, 10)
        outcome = 'failed' if fails else 'completed'
        assert get_session_admission(db, admission_id=receipt.admission_id)['outcome'] == outcome
        complete = [f['params']['payload'] for f in frames if f['params']['type'] == 'message.complete']
        assert complete[-1]['outcome'] == outcome
        assert complete[-1]['status'] == ('error' if fails else 'complete')
    finally:
        db.close()
