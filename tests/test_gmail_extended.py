"""Extended Gmail adapter tests — edge cases beyond the 7 required tests.

These cover:
  - Malformed Pub/Sub envelopes
  - HTML-only emails (no text/plain)
  - Multiple attachments
  - Unicode content
  - Empty body
  - Display name extraction
  - Artifact store security (path traversal)
  - Thread continuity in replies
"""

from __future__ import annotations

import base64
import json
from datetime import datetime
from email.message import EmailMessage

import pytest

from glc.channels.catalogue.gmail.adapter import Adapter
from glc.channels.catalogue.gmail.artifacts import (
    _validate_ref,
    get,
    remove,
    store,
)
from glc.channels.envelope import ChannelMessage, ChannelReply
from glc.security.pairing import get_pairing_store
from tests.channels.mocks.gmail_mock import OWNER_ID, GmailMock


@pytest.fixture
def mock():
    return GmailMock()


@pytest.fixture
def pair_owner():
    s = get_pairing_store()
    s.force_pair_owner("gmail", OWNER_ID, user_handle="owner")
    yield
    s.revoke("gmail", OWNER_ID)


# ─── Malformed Input ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_malformed_pubsub_missing_message_key(mock):
    adapter = Adapter(config={"mock": mock})
    with pytest.raises(ValueError, match="Malformed Pub/Sub envelope"):
        await adapter.on_message({"not_message": {}})


@pytest.mark.asyncio
async def test_malformed_pubsub_invalid_base64(mock):
    adapter = Adapter(config={"mock": mock})
    envelope = {"message": {"data": "not-valid-base64!!!"}}
    with pytest.raises(ValueError, match="Malformed Pub/Sub envelope"):
        await adapter.on_message(envelope)


@pytest.mark.asyncio
async def test_malformed_pubsub_missing_history_id(mock):
    adapter = Adapter(config={"mock": mock})
    data = base64.b64encode(json.dumps({"emailAddress": "x@y.com"}).encode()).decode()
    envelope = {"message": {"data": data}}
    with pytest.raises(ValueError, match="Malformed Pub/Sub envelope"):
        await adapter.on_message(envelope)


# ─── HTML-only Email ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_html_only_email_returns_empty_text(mock, pair_owner):
    """An email with only text/html (no text/plain) should yield empty text."""
    msg = EmailMessage()
    msg["From"] = OWNER_ID
    msg["To"] = "bot@example.com"
    msg["Subject"] = "HTML only"
    msg.set_content("<h1>Hello</h1><p>World</p>", subtype="html")

    raw_bytes = bytes(msg)
    raw_b64 = base64.urlsafe_b64encode(raw_bytes).decode().rstrip("=")

    msg_id = "msg-html-only"
    history_id = 9999
    mock._messages[msg_id] = {
        "id": msg_id,
        "threadId": "thread-html",
        "raw": raw_b64,
    }
    mock._history[history_id] = [msg_id]

    inner = json.dumps({"emailAddress": "bot@example.com", "historyId": history_id})
    data = base64.b64encode(inner.encode()).decode()
    envelope = {"message": {"data": data}}

    adapter = Adapter(config={"mock": mock})
    result = await adapter.on_message(envelope)

    assert result is not None
    assert result.text == ""


# ─── Unicode Content ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unicode_body_preserved(mock, pair_owner):
    """Unicode characters in email body are preserved."""
    unicode_text = "Meeting at 3pm ☕ 你好 مرحبا"

    msg = EmailMessage()
    msg["From"] = OWNER_ID
    msg["To"] = "bot@example.com"
    msg["Subject"] = "Unicode"
    msg.set_content(unicode_text)

    raw_bytes = bytes(msg)
    raw_b64 = base64.urlsafe_b64encode(raw_bytes).decode().rstrip("=")

    msg_id = "msg-unicode"
    history_id = 8888
    mock._messages[msg_id] = {"id": msg_id, "threadId": "thread-uni", "raw": raw_b64}
    mock._history[history_id] = [msg_id]

    inner = json.dumps({"emailAddress": "bot@example.com", "historyId": history_id})
    data = base64.b64encode(inner.encode()).decode()
    envelope = {"message": {"data": data}}

    adapter = Adapter(config={"mock": mock})
    result = await adapter.on_message(envelope)

    assert result is not None
    assert "☕" in result.text
    assert "你好" in result.text


# ─── Empty Body ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_empty_body_no_crash(mock, pair_owner):
    """Email with empty body doesn't crash the adapter."""
    msg = EmailMessage()
    msg["From"] = OWNER_ID
    msg["To"] = "bot@example.com"
    msg["Subject"] = "No body"
    msg.set_content("")

    raw_bytes = bytes(msg)
    raw_b64 = base64.urlsafe_b64encode(raw_bytes).decode().rstrip("=")

    msg_id = "msg-empty"
    history_id = 7777
    mock._messages[msg_id] = {"id": msg_id, "threadId": "thread-empty", "raw": raw_b64}
    mock._history[history_id] = [msg_id]

    inner = json.dumps({"emailAddress": "bot@example.com", "historyId": history_id})
    data = base64.b64encode(inner.encode()).decode()
    envelope = {"message": {"data": data}}

    adapter = Adapter(config={"mock": mock})
    result = await adapter.on_message(envelope)

    assert result is not None
    assert result.text is not None  # empty string, not None


# ─── Display Name Extraction ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_display_name_stripped_for_trust(mock, pair_owner):
    """'Display Name <email>' format is stripped to bare email for trust lookup."""
    msg = EmailMessage()
    msg["From"] = f"Owner Person <{OWNER_ID}>"
    msg["To"] = "bot@example.com"
    msg["Subject"] = "Display name test"
    msg.set_content("hello")

    raw_bytes = bytes(msg)
    raw_b64 = base64.urlsafe_b64encode(raw_bytes).decode().rstrip("=")

    msg_id = "msg-display"
    history_id = 6666
    mock._messages[msg_id] = {"id": msg_id, "threadId": "thread-display", "raw": raw_b64}
    mock._history[history_id] = [msg_id]

    inner = json.dumps({"emailAddress": "bot@example.com", "historyId": history_id})
    data = base64.b64encode(inner.encode()).decode()
    envelope = {"message": {"data": data}}

    adapter = Adapter(config={"mock": mock})
    result = await adapter.on_message(envelope)

    assert result is not None
    assert result.channel_user_id == OWNER_ID
    assert result.trust_level == "owner_paired"


# ─── Attachments ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pdf_attachment_extracted(mock, pair_owner):
    """A PDF attachment produces Attachment(kind='file', mime='application/pdf')."""
    msg = EmailMessage()
    msg["From"] = OWNER_ID
    msg["To"] = "bot@example.com"
    msg["Subject"] = "PDF attached"
    msg.set_content("See attached PDF")
    msg.add_attachment(
        b"%PDF-1.4 fake pdf content here",
        maintype="application",
        subtype="pdf",
        filename="report.pdf",
    )

    raw_bytes = bytes(msg)
    raw_b64 = base64.urlsafe_b64encode(raw_bytes).decode().rstrip("=")

    msg_id = "msg-pdf"
    history_id = 5555
    mock._messages[msg_id] = {"id": msg_id, "threadId": "thread-pdf", "raw": raw_b64}
    mock._history[history_id] = [msg_id]

    inner = json.dumps({"emailAddress": "bot@example.com", "historyId": history_id})
    data = base64.b64encode(inner.encode()).decode()
    envelope = {"message": {"data": data}}

    adapter = Adapter(config={"mock": mock})
    result = await adapter.on_message(envelope)

    assert result is not None
    assert result.text is not None
    assert "See attached PDF" in result.text
    assert len(result.attachments) == 1
    assert result.attachments[0].kind == "file"
    assert result.attachments[0].mime == "application/pdf"
    assert result.attachments[0].metadata["filename"] == "report.pdf"
    assert result.attachments[0].ref.startswith("art:")


@pytest.mark.asyncio
async def test_multiple_attachments(mock, pair_owner):
    """Multiple attachments are all extracted."""
    msg = EmailMessage()
    msg["From"] = OWNER_ID
    msg["To"] = "bot@example.com"
    msg["Subject"] = "Multiple files"
    msg.set_content("Three files attached")
    msg.add_attachment(b"PNG fake", maintype="image", subtype="png", filename="photo.png")
    msg.add_attachment(b"PDF fake", maintype="application", subtype="pdf", filename="doc.pdf")
    msg.add_attachment(b"MP3 fake", maintype="audio", subtype="mpeg", filename="song.mp3")

    raw_bytes = bytes(msg)
    raw_b64 = base64.urlsafe_b64encode(raw_bytes).decode().rstrip("=")

    msg_id = "msg-multi"
    history_id = 4444
    mock._messages[msg_id] = {"id": msg_id, "threadId": "thread-multi", "raw": raw_b64}
    mock._history[history_id] = [msg_id]

    inner = json.dumps({"emailAddress": "bot@example.com", "historyId": history_id})
    data = base64.b64encode(inner.encode()).decode()
    envelope = {"message": {"data": data}}

    adapter = Adapter(config={"mock": mock})
    result = await adapter.on_message(envelope)

    assert result is not None
    assert len(result.attachments) == 3
    kinds = {a.kind for a in result.attachments}
    assert kinds == {"image", "file", "audio"}


# ─── Thread Continuity in Reply ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reply_preserves_thread_id(mock, pair_owner):
    """send() includes threadId in payload when thread_id is set."""
    adapter = Adapter(config={"mock": mock})
    reply = ChannelReply(
        channel="gmail",
        channel_user_id=OWNER_ID,
        text="replying",
        thread_id="thread-original-123",
    )
    result = await adapter.send(reply)
    assert len(mock.send_log) == 1
    payload = mock.send_log[0]
    assert payload["threadId"] == "thread-original-123"


@pytest.mark.asyncio
async def test_reply_without_thread_id_omits_field(mock, pair_owner):
    """send() does NOT include threadId when thread_id is None."""
    adapter = Adapter(config={"mock": mock})
    reply = ChannelReply(
        channel="gmail",
        channel_user_id=OWNER_ID,
        text="new conversation",
    )
    result = await adapter.send(reply)
    payload = mock.send_log[0]
    assert "threadId" not in payload


# ─── Artifact Store Security ─────────────────────────────────────────────────


def test_artifact_path_traversal_blocked():
    """Attempting path traversal via art: ref returns None."""
    assert _validate_ref("art:../../etc/passwd") is None
    assert _validate_ref("art:../../../secrets") is None
    assert _validate_ref("art:hello world") is None
    assert _validate_ref("art:") is None
    assert get("art:../../etc/passwd") is None


def test_artifact_valid_ref():
    """Valid 16-char hex refs pass validation."""
    assert _validate_ref("art:0123456789abcdef") == "0123456789abcdef"
    assert _validate_ref("art:deadbeefcafebabe") == "deadbeefcafebabe"


def test_artifact_store_and_retrieve():
    """Store bytes, retrieve them, then cleanup."""
    data = b"test artifact content"
    ref = store(data, filename="test.txt")
    assert ref.startswith("art:")
    assert get(ref) == data
    assert remove(ref) is True
    assert get(ref) is None


# ─── Reply Format ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reply_has_in_reply_to_header(mock, pair_owner):
    """Reply with thread_id produces In-Reply-To and References headers."""
    adapter = Adapter(config={"mock": mock})
    reply = ChannelReply(
        channel="gmail",
        channel_user_id=OWNER_ID,
        text="thanks",
        thread_id="msg-id-original",
    )
    await adapter.send(reply)
    payload = mock.send_log[0]
    raw_b64 = payload["raw"]
    padded = raw_b64 + "=" * (-len(raw_b64) % 4)
    decoded = base64.urlsafe_b64decode(padded.encode())
    assert b"In-Reply-To: msg-id-original" in decoded
    assert b"References: msg-id-original" in decoded


# ─── SPF/DKIM/DMARC Spoofing Protection ──────────────────────────────────────


async def _seed_message_with_auth(mock, from_addr: str, auth_header: str | None) -> dict:
    """Seed a mock message with a given From: and Authentication-Results header."""
    msg = EmailMessage()
    msg["From"] = from_addr
    msg["To"] = "bot@example.com"
    msg["Subject"] = "test"
    if auth_header is not None:
        msg["Authentication-Results"] = auth_header
    msg.set_content("hello")

    raw_bytes = bytes(msg)
    raw_b64 = base64.urlsafe_b64encode(raw_bytes).decode().rstrip("=")

    msg_id = f"msg-auth-{hash(from_addr + str(auth_header)) & 0xFFFF}"
    history_id = 3333 + (hash(auth_header or "") & 0xFF)
    mock._messages[msg_id] = {"id": msg_id, "threadId": f"thread-{msg_id}", "raw": raw_b64}
    mock._history[history_id] = [msg_id]

    inner = json.dumps({"emailAddress": "bot@example.com", "historyId": history_id})
    data = base64.b64encode(inner.encode()).decode()
    return {"message": {"data": data}}


@pytest.mark.asyncio
async def test_spoofed_from_downgraded_to_untrusted(mock, pair_owner):
    """Attacker spoofs the owner's From: address with SPF/DKIM/DMARC failing.

    Without the fix: adapter would classify as owner_paired.
    With the fix: authentication check downgrades to untrusted.
    """
    envelope = await _seed_message_with_auth(
        mock,
        from_addr=OWNER_ID,
        auth_header="mx.google.com; spf=fail smtp.mailfrom=attacker.com; dkim=none; dmarc=fail header.from=example.com",
    )
    adapter = Adapter(config={"mock": mock, "require_sender_auth": True})
    result = await adapter.on_message(envelope)
    assert result is not None
    assert result.channel_user_id == OWNER_ID
    assert result.trust_level == "untrusted", "spoofed sender must not be trusted"


@pytest.mark.asyncio
async def test_authenticated_owner_stays_owner_paired(mock, pair_owner):
    """Legitimate owner email with SPF/DKIM/DMARC all passing keeps owner_paired."""
    envelope = await _seed_message_with_auth(
        mock,
        from_addr=OWNER_ID,
        auth_header="mx.google.com; spf=pass smtp.mailfrom=example.com; dkim=pass header.d=example.com; dmarc=pass header.from=example.com",
    )
    adapter = Adapter(config={"mock": mock, "require_sender_auth": True})
    result = await adapter.on_message(envelope)
    assert result is not None
    assert result.trust_level == "owner_paired"


@pytest.mark.asyncio
async def test_partial_auth_pass_downgrades(mock, pair_owner):
    """SPF pass but DKIM fail should still downgrade — we require ALL three."""
    envelope = await _seed_message_with_auth(
        mock,
        from_addr=OWNER_ID,
        auth_header="mx.google.com; spf=pass; dkim=fail; dmarc=fail",
    )
    adapter = Adapter(config={"mock": mock, "require_sender_auth": True})
    result = await adapter.on_message(envelope)
    assert result is not None
    assert result.trust_level == "untrusted"


@pytest.mark.asyncio
async def test_missing_auth_header_fails_in_strict_mode(mock, pair_owner):
    """No Authentication-Results header in strict mode → untrusted."""
    envelope = await _seed_message_with_auth(
        mock,
        from_addr=OWNER_ID,
        auth_header=None,
    )
    adapter = Adapter(config={"mock": mock, "require_sender_auth": True})
    result = await adapter.on_message(envelope)
    assert result is not None
    assert result.trust_level == "untrusted"


@pytest.mark.asyncio
async def test_missing_auth_header_permissive_in_test_mode(mock, pair_owner):
    """No Authentication-Results header in permissive mode (default) → trust preserved.

    This keeps the existing CI tests passing since the mock doesn't add
    the Authentication-Results header.
    """
    envelope = await _seed_message_with_auth(
        mock,
        from_addr=OWNER_ID,
        auth_header=None,
    )
    adapter = Adapter(config={"mock": mock})  # require_sender_auth defaults False
    result = await adapter.on_message(envelope)
    assert result is not None
    assert result.trust_level == "owner_paired"
