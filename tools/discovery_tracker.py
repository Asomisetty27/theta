"""
Discovery call outreach tracker.

Lightweight JSONL-based CRM for tracking cold outreach to early adopters.
Stored at ~/.theta/outreach.jsonl — one record per contact, updated in place.

Usage:
  python tools/discovery_tracker.py add --org "Example RC" --name "Jane Doe" \
      --email "jane@example.edu" --notes "HPC training lead, 300 H100 cluster"

  python tools/discovery_tracker.py list
  python tools/discovery_tracker.py list --status sent

  python tools/discovery_tracker.py update --id <id> --status replied
  python tools/discovery_tracker.py update --id <id> --status called --notes "Interested, wants a demo"

  python tools/discovery_tracker.py followup   # list contacts needing follow-up (sent > 5 days)
  python tools/discovery_tracker.py stats      # summary counts by status

Statuses: draft | sent | replied | scheduled | called | declined | converted
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

OUTREACH_FILE = Path.home() / ".theta" / "outreach.jsonl"
FOLLOWUP_DAYS = 5   # days before a 'sent' contact needs a nudge

STATUSES = {"draft", "sent", "replied", "scheduled", "called", "declined", "converted"}

STATUS_COLOR = {
    "draft":     "\033[90m",  # grey
    "sent":      "\033[33m",  # yellow
    "replied":   "\033[36m",  # cyan
    "scheduled": "\033[34m",  # blue
    "called":    "\033[32m",  # green
    "declined":  "\033[31m",  # red
    "converted": "\033[35m",  # magenta
}
RESET = "\033[0m"


def _load() -> dict[str, dict]:
    if not OUTREACH_FILE.exists():
        return {}
    records = {}
    for line in OUTREACH_FILE.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            records[r["id"]] = r
        except (json.JSONDecodeError, KeyError):
            continue
    return records


def _save(records: dict[str, dict]) -> None:
    OUTREACH_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTREACH_FILE.write_text(
        "\n".join(json.dumps(r) for r in records.values()) + "\n"
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _make_id(org: str, email: str) -> str:
    raw = f"{org.lower().strip()}:{email.lower().strip()}"
    return hashlib.sha1(raw.encode()).hexdigest()[:8]


def _days_since(iso: str) -> float:
    try:
        dt = datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds() / 86400
    except Exception:
        return 0.0


def cmd_add(args: argparse.Namespace) -> None:
    records = _load()
    contact_id = _make_id(args.org, args.email)
    if contact_id in records:
        print(f"Contact already exists (id={contact_id}). Use 'update' to change status.")
        return
    record = {
        "id":      contact_id,
        "org":     args.org,
        "name":    args.name or "",
        "email":   args.email,
        "segment": args.segment or "",
        "status":  "sent" if args.status == "sent" else "draft",
        "sent_at": _now_iso() if args.status == "sent" else None,
        "notes":   args.notes or "",
        "created": _now_iso(),
        "updated": _now_iso(),
        "history": [],
    }
    records[contact_id] = record
    _save(records)
    print(f"Added: [{contact_id}] {args.org} — {args.email} ({record['status']})")


def cmd_update(args: argparse.Namespace) -> None:
    records = _load()
    if args.id not in records:
        print(f"No contact with id={args.id}. Use 'list' to find ids.")
        sys.exit(1)
    r = records[args.id]
    old_status = r["status"]
    if args.status:
        if args.status not in STATUSES:
            print(f"Invalid status. Must be one of: {', '.join(sorted(STATUSES))}")
            sys.exit(1)
        r["history"].append({"ts": _now_iso(), "from": old_status, "to": args.status})
        r["status"] = args.status
        if args.status == "sent" and not r.get("sent_at"):
            r["sent_at"] = _now_iso()
    if args.notes:
        r["notes"] = args.notes
    r["updated"] = _now_iso()
    _save(records)
    print(f"Updated [{args.id}] {r['org']}: {old_status} → {r['status']}")


def cmd_list(args: argparse.Namespace) -> None:
    records = _load()
    if not records:
        print("No contacts tracked yet. Use 'add' to add one.")
        return
    filtered = list(records.values())
    if args.status:
        filtered = [r for r in filtered if r["status"] == args.status]
    filtered.sort(key=lambda r: r.get("updated", ""), reverse=True)
    print(f"\n{'ID':8}  {'STATUS':10}  {'ORG':28}  {'NAME':20}  EMAIL")
    print("-" * 95)
    for r in filtered:
        color = STATUS_COLOR.get(r["status"], "")
        days = ""
        if r["status"] == "sent" and r.get("sent_at"):
            d = _days_since(r["sent_at"])
            days = f" ({d:.0f}d)"
        print(
            f"{color}{r['id']:8}{RESET}  "
            f"{color}{r['status']:10}{RESET}  "
            f"{r['org'][:28]:28}  "
            f"{r['name'][:20]:20}  "
            f"{r['email']}{days}"
        )
        if r.get("notes"):
            print(f"          └─ {r['notes']}")
    print(f"\n{len(filtered)} contact(s).")


def cmd_followup(args: argparse.Namespace) -> None:
    records = _load()
    needs_followup = [
        r for r in records.values()
        if r["status"] == "sent"
        and r.get("sent_at")
        and _days_since(r["sent_at"]) >= FOLLOWUP_DAYS
    ]
    if not needs_followup:
        print(f"No contacts waiting > {FOLLOWUP_DAYS} days. All good.")
        return
    print(f"\n{len(needs_followup)} contact(s) need follow-up:\n")
    for r in sorted(needs_followup, key=lambda x: _days_since(x.get("sent_at", "")), reverse=True):
        days = _days_since(r["sent_at"])
        print(f"  [{r['id']}] {r['org']} — {r['name']} <{r['email']}>")
        print(f"        Sent {days:.0f} days ago. Notes: {r.get('notes', 'none')}")
        print(f"        → Update: python tools/discovery_tracker.py update --id {r['id']} --status replied")
        print()


def cmd_stats(args: argparse.Namespace) -> None:
    records = _load()
    if not records:
        print("No contacts yet.")
        return
    counts: dict[str, int] = {}
    for r in records.values():
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    total = sum(counts.values())
    print(f"\nOutreach summary ({total} total):")
    for status in ["draft", "sent", "replied", "scheduled", "called", "declined", "converted"]:
        n = counts.get(status, 0)
        if n:
            color = STATUS_COLOR.get(status, "")
            bar = "█" * n
            print(f"  {color}{status:12}{RESET}  {bar}  {n}")
    converted = counts.get("converted", 0)
    sent = sum(counts.get(s, 0) for s in ["sent", "replied", "scheduled", "called", "converted"])
    if sent > 0:
        print(f"\n  Conversion rate: {converted}/{sent} = {100*converted/sent:.0f}%")


def main() -> None:
    p = argparse.ArgumentParser(description="Theta discovery outreach tracker")
    sub = p.add_subparsers(dest="cmd", required=True)

    add_p = sub.add_parser("add", help="Add a new contact")
    add_p.add_argument("--org",     required=True, help="Organization name")
    add_p.add_argument("--email",   required=True, help="Contact email")
    add_p.add_argument("--name",    help="Contact name")
    add_p.add_argument("--segment", help="Segment (hpc / crypto-ai / runpod-host / eleutherai)")
    add_p.add_argument("--status",  default="sent", choices=sorted(STATUSES))
    add_p.add_argument("--notes",   help="Free-form notes about the contact")

    upd_p = sub.add_parser("update", help="Update contact status or notes")
    upd_p.add_argument("--id",     required=True, help="Contact ID (8 hex chars)")
    upd_p.add_argument("--status", choices=sorted(STATUSES))
    upd_p.add_argument("--notes",  help="Update notes")

    list_p = sub.add_parser("list", help="List all contacts")
    list_p.add_argument("--status", choices=sorted(STATUSES), help="Filter by status")

    sub.add_parser("followup", help="List contacts needing follow-up (sent > 5 days)")
    sub.add_parser("stats",    help="Summary counts by status")

    args = p.parse_args()
    {"add": cmd_add, "update": cmd_update, "list": cmd_list,
     "followup": cmd_followup, "stats": cmd_stats}[args.cmd](args)


if __name__ == "__main__":
    main()
