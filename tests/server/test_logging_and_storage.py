"""What production writes to its log, and whether it can tell that its storage
is about to be lost.

On Railway the server logs one JSON object per line (LOG_FORMAT=json, the
default there): `level` drives the log colour and @level: filter, `message` is
the text, every other key is a filterable attribute.
"""
import json
import os

from conftest import detection, post, post_det


def records(out):
    """Every line of captured stdout as a JSON record (fails on a stray line)."""
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def events(out, name):
    return [r for r in records(out) if r.get('event') == name]


# ── Format ────────────────────────────────────────────────────────────────────
def test_json_mode_one_record_per_line_with_level_component_and_fields(harness, capsys):
    srv = harness.start(LOG_FORMAT='json')
    client = srv.app.test_client()
    capsys.readouterr()
    assert post_det(client).status_code == 200
    [det] = events(capsys.readouterr().out, 'detection')
    assert det['level'] == 'info' and det['component'] == 'Det'
    assert det['message'].startswith('[Det #1] ')
    assert det['det_id'] == 1 and det['species'] == 'aedes_albopictus'
    assert det['conf'] == 0.83 and det['located'] is True


def test_json_mode_keeps_client_input_inside_its_record(harness, capsys):
    srv = harness.start(LOG_FORMAT='json')
    client = srv.app.test_client()
    capsys.readouterr()
    forged = 'x\n{"level":"info","message":"forged"}'
    assert post_det(client, species=forged).status_code == 400
    recs = records(capsys.readouterr().out)          # every line parses as a record
    assert not any(r['message'] == 'forged' for r in recs)
    [rej] = [r for r in recs if r.get('event') == 'rejected_detection']
    assert rej['level'] == 'warn' and rej['component'] == 'Det'


def test_text_mode_prints_plain_lines(harness, capsys):
    srv = harness.start(LOG_FORMAT='text')
    capsys.readouterr()
    post_det(srv.app.test_client())
    out = capsys.readouterr().out
    assert out.startswith('[Det #1] Aedes albopictus conf=0.830')
    assert '{' not in out


def test_format_defaults_to_json_on_railway_only(harness):
    assert harness.start(LOG_FORMAT=None).LOG_FORMAT == 'text'
    assert harness.start(LOG_FORMAT='xml').LOG_FORMAT == 'text'      # invalid → default
    assert harness.start(LOG_FORMAT=None, RAILWAY_ENVIRONMENT_ID='env').LOG_FORMAT == 'json'


# ── What gets logged ──────────────────────────────────────────────────────────
def test_boot_record_identifies_the_build_and_its_storage(harness, capsys):
    srv = harness.start(LOG_FORMAT='json', RAILWAY_GIT_COMMIT_SHA='0123456789abcdef0123',
                        RAILWAY_DEPLOYMENT_ID='dep-1')
    [boot] = events(capsys.readouterr().out, 'boot')
    assert boot['level'] == 'info' and boot['component'] == 'Boot'
    assert boot['commit'] == '0123456789ab' and boot['deployment'] == 'dep-1'
    assert boot['data_dir'] == harness.data_dir and boot['pid'] == os.getpid()
    assert boot['ephemeral'] is (srv.EPHEMERAL_REASON is not None)


def test_ephemeral_storage_is_an_error_record(harness, capsys):
    srv = harness.start(LOG_FORMAT='json', RAILWAY_ENVIRONMENT_ID='env',
                        RAILWAY_VOLUME_MOUNT_PATH=None)
    assert srv.EPHEMERAL_REASON          # tmp dir, and no Railway volume
    [rec] = events(capsys.readouterr().out, 'ephemeral_storage')
    assert rec['level'] == 'error' and rec['reason'] == srv.EPHEMERAL_REASON
    assert 'PERMANENTLY DELETE' in rec['message']


def test_running_as_root_on_a_platform_is_flagged(harness, capsys):
    harness.start(LOG_FORMAT='json', RAILWAY_ENVIRONMENT_ID='env')
    flagged = events(capsys.readouterr().out, 'running_as_root')
    assert len(flagged) == (1 if os.getuid() == 0 else 0)


def test_internal_error_logs_a_stack_without_the_exception_text(harness, capsys, monkeypatch):
    srv = harness.start(LOG_FORMAT='json')

    detail = 'secret detail\n{"level":"info","message":"forged"}'

    def boom():
        raise RuntimeError(detail)        # the text must not appear in the stack
    monkeypatch.setattr(srv, 'full_stats', boom)
    capsys.readouterr()
    assert srv.app.test_client().get('/federated/stats').status_code == 500
    recs = records(capsys.readouterr().out)
    [rec] = [r for r in recs if r.get('event') == 'internal_error']
    assert rec['level'] == 'error' and rec['exc_type'] == 'RuntimeError'
    assert 'in boom' in rec['stack'] and 'in get_stats' in rec['stack']
    assert 'secret detail' not in rec['stack']
    assert not any(r['message'] == 'forged' for r in recs)


def test_stats_record_reports_change_since_the_previous_one(harness, capsys):
    srv = harness.start(LOG_FORMAT='json')
    client = srv.app.test_client()
    post_det(client, deviceId='a' * 32)
    post_det(client, deviceId='b' * 32)  # same device+species would merge into one find
    assert post(client, '/detection', detection(), origin='https://evil.example').status_code == 403
    capsys.readouterr()
    srv._log_stats()
    [s1] = events(capsys.readouterr().out, 'stats')
    assert s1['level'] == 'info' and s1['component'] == 'Stats'
    assert (s1['total_detections'], s1['new_detections'], s1['rejected']) == (2, 2, 1)

    post_det(client, deviceId='c' * 32)
    srv._write_failing = True              # e.g. the disk filled up
    capsys.readouterr()
    srv._log_stats()
    [s2] = events(capsys.readouterr().out, 'stats')
    assert (s2['total_detections'], s2['new_detections'], s2['rejected']) == (3, 1, 0)
    assert s2['level'] == 'error'


def test_exit_record_carries_the_final_counters(harness, capsys):
    srv = harness.start(LOG_FORMAT='json')
    post_det(srv.app.test_client())
    capsys.readouterr()
    harness.restart(srv, graceful=True)
    [done] = events(capsys.readouterr().out, 'exit')
    assert done['message'] == '[Exit] done'
    assert done['total_detections'] == 1 and done['write_errors'] == 0


# ── Storage check ─────────────────────────────────────────────────────────────
def test_storage_check_flags_each_way_data_is_lost(srv, monkeypatch):
    for k in ('RAILWAY_ENVIRONMENT_ID', 'RAILWAY_PROJECT_ID', 'RAILWAY_VOLUME_MOUNT_PATH'):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(srv, '_in_container', lambda: False)
    assert srv._ephemeral_reason('/tmp') == 'DATA_DIR is under /tmp'
    assert srv._ephemeral_reason('/tmp/mn') == 'DATA_DIR is under /tmp'
    assert srv._ephemeral_reason('/srv/mn') is None             # a plain host disk

    # Railway with no volume — how production ran, while the old check said "persistent".
    monkeypatch.setenv('RAILWAY_ENVIRONMENT_ID', 'env')
    assert srv._ephemeral_reason('/data') == 'no Railway volume is attached to this service'
    monkeypatch.setenv('RAILWAY_VOLUME_MOUNT_PATH', '/srv/vol')
    assert srv._ephemeral_reason('/srv/vol') is None
    assert srv._ephemeral_reason('/srv/vol/data') is None
    assert 'not on the Railway volume' in srv._ephemeral_reason('/srv/vol-other')
    assert 'not on the Railway volume' in srv._ephemeral_reason('/data')

    # Any other container: the directory must be on a mounted volume.
    for k in ('RAILWAY_ENVIRONMENT_ID', 'RAILWAY_VOLUME_MOUNT_PATH'):
        monkeypatch.delenv(k)
    monkeypatch.setattr(srv, '_in_container', lambda: True)
    monkeypatch.setattr(srv, '_on_mounted_volume', lambda p: False)
    assert srv._ephemeral_reason('/data') == 'DATA_DIR is on the container filesystem (no volume mounted)'
    monkeypatch.setattr(srv, '_on_mounted_volume', lambda p: True)
    assert srv._ephemeral_reason('/data') is None


def test_mount_detection(srv):
    assert srv._on_mounted_volume('/') is False
    if os.path.ismount('/proc'):
        assert srv._on_mounted_volume('/proc/self') is True


def test_health_says_why_storage_is_ephemeral(client, srv):
    h = client.get('/health').get_json()
    assert h['ephemeral'] is (srv.EPHEMERAL_REASON is not None)
    assert h['ephemeral_reason'] == srv.EPHEMERAL_REASON
