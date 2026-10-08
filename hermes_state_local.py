"""Private local creation receipts in the canonical runtime transaction owner."""
import json
import time

from hermes_state_runtime import RuntimeStoreError, _epoch, _json

POLICY_PREFIX = 'gateway.local_policy.v1:'
API_BINDING_PREFIX = 'gateway.api.binding.v1.'
API_DECLARED_PREFIX = 'gateway.api.conversation.v1.'


def commit_local_session(db, *, epoch, receipt):
    """Reserve the row, route and immutable policy together, before publication."""
    encoded = _json(receipt)
    receipt = json.loads(encoded)
    sid, route = receipt['session_id'], receipt['route']
    key = POLICY_PREFIX + sid
    def write(conn):
        _epoch(conn, epoch)
        old = conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()
        if old is not None:
            try:
                saved = json.loads(old[0])
                same = all(saved[k] == receipt[k] for k in (
                    'profile_id', 'principal_id', 'request_id', 'session_id', 'route'))
                same = same and saved['policy']['request_json'] == receipt['policy']['request_json']
            except (ValueError, TypeError, KeyError) as exc:
                raise RuntimeStoreError('storage_unavailable') from exc
            if not same:
                raise RuntimeStoreError('invalid_params')
            return saved
        from hermes_state_mutation_retirement import RETIRED_PREFIX
        if conn.execute('SELECT 1 FROM state_meta WHERE key=?', (RETIRED_PREFIX + sid,)).fetchone():
            # Deleted: its receipt was retired with the transcript; a create/cron retry must not
            # resurrect the row (and re-fire its prompt) under the same deterministic id.
            raise RuntimeStoreError('not_found')
        if ('legacy_session_id' not in receipt and
                conn.execute('SELECT 1 FROM sessions WHERE id=?', (sid,)).fetchone()):
            raise RuntimeStoreError('storage_unavailable')
        if conn.execute("SELECT 1 FROM gateway_routing WHERE scope='' AND session_key=?", (route,)).fetchone():
            raise RuntimeStoreError('admission_conflict')
        if 'legacy_session_id' in receipt:
            from hermes_state_local_migration import bind_legacy_target
            bind_legacy_target(db, conn, receipt)
        policy, entry = receipt['policy'], receipt['entry']
        from datetime import datetime
        started = datetime.fromisoformat(entry['created_at']).timestamp()
        conn.execute('''INSERT INTO sessions(id,source,user_id,session_key,chat_id,chat_type,
            model,cwd,profile_name,origin_json,started_at) VALUES(?,?,?,?,?,'dm',?,?,?,?,?)
            ON CONFLICT(id) DO NOTHING''',
            (sid, policy['source'], receipt['principal_id'], route, entry['origin']['chat_id'],
             policy['model'], policy['cwd'], db._own_profile_name(), _json(entry['origin']), started))
        conn.execute("INSERT INTO gateway_routing(scope,session_key,entry_json,updated_at) VALUES('',?,?,?)",
                     (route, _json(entry), started))
        conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)', (key, encoded))
        return receipt
    return db._execute_write(write)


def retire_local_receipts(conn, session_ids):
    """Deletion fence half for owner receipts. A local creation policy carries the launch request
    (``request_json``: a cron fire's job and raw prompt), so it goes exactly when its route does:
    when the receipt's current physical target is deleted (a pruned reset ancestor whose logical
    session lives on in a child keeps it). An API binding and the declared conversation key that
    resolves to it go with their id, so that key starts a fresh session."""
    targets = set(session_ids)
    if not targets:
        return
    # Prefix scans (GLOB uses the key index), not one LIKE per id: a prune retires thousands.
    gone = [key for key, value in conn.execute('SELECT key,value FROM state_meta WHERE key GLOB ?',
                                               (POLICY_PREFIX + '*',))
            if json.loads(value).get('entry', {}).get('session_id') in targets]
    gone += [key for key, value in conn.execute('SELECT key,value FROM state_meta WHERE key GLOB ?',
                                                (API_DECLARED_PREFIX + '*',)) if value in targets]
    gone += [API_BINDING_PREFIX + sid for sid in targets]
    conn.executemany('DELETE FROM state_meta WHERE key=?', [(key,) for key in gone])


def local_receipt(db, session_id):
    with db._read_ctx() as conn:
        row = conn.execute('SELECT value FROM state_meta WHERE key=?', (POLICY_PREFIX + session_id,)).fetchone()
    if row is None:
        raise RuntimeStoreError('storage_unavailable')
    try:
        value = json.loads(row[0])
        if not isinstance(value, dict):
            raise ValueError('invalid receipt')
        return value
    except (TypeError, ValueError) as exc:
        raise RuntimeStoreError('storage_unavailable') from exc


def end_idle_local_session(db, *, epoch, session_id, target_id, reason):
    """Stamp ``ended_at`` on *target_id* only while the logical *session_id*'s FIFO is idle.

    The idle check and the stamp share one owner write txn, so an admission committed first keeps
    the row open; one committed after is reopened by the drain (``reopen_local_session``)."""
    def write(conn):
        _epoch(conn, epoch)
        if conn.execute("SELECT 1 FROM session_admissions WHERE target_session_id=? AND status!='terminal'",
                        (session_id,)).fetchone():
            return 0
        if conn.execute("SELECT 1 FROM worker_executions WHERE session_id IN (?,?) AND status!='terminal'",
                        (session_id, target_id)).fetchone():
            return 0
        return db._end_and_bump(conn, 'UPDATE sessions SET ended_at=?, end_reason=? WHERE id=? AND ended_at IS NULL',
                                (time.time(), reason, target_id), target_id, reason)
    return db._execute_write(write)
