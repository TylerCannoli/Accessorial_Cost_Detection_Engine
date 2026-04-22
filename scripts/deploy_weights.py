"""
scripts/deploy_weights.py
=========================
Transfers trained model weights from the university server to your
local machine (or any target host running Streamlit).

The weights are gitignored so they cannot be pushed via git —
this script uses SCP/SFTP to copy them directly.

Usage:
    # Pull weights from server to local models/ directory
    python scripts/deploy_weights.py --pull

    # Push local weights up to a remote host
    python scripts/deploy_weights.py --push --host myhost.com --user myuser

    # Just check what files exist on both ends
    python scripts/deploy_weights.py --status

Requires:
    pip install paramiko scp

Configure server details via .env or command-line flags:
    DEPLOY_HOST     — server hostname (e.g. aimlsrv.walton.uark.edu)
    DEPLOY_USER     — SSH username
    DEPLOY_KEY      — path to SSH private key (optional, uses password if not set)
    DEPLOY_REMOTE_DIR — remote path to PACE repo root
"""

import argparse
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

MODEL_FILES = [
    "models/pace_transformer_ltl.pt",
    "models/artifacts_ltl.pkl",
    "models/pace_transformer_ftl.pt",
    "models/artifacts_ftl.pkl",
]

LOCAL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_ssh_client(host: str, user: str, key_path: str = None, password: str = None):
    try:
        import paramiko
    except ImportError:
        print("ERROR: paramiko not installed. Run: pip install paramiko scp")
        sys.exit(1)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # nosec B507

    if key_path and os.path.exists(key_path):
        client.connect(hostname=host, username=user, key_filename=key_path)
        print(f"  Connected to {host} using key {key_path}")
    elif password:
        client.connect(hostname=host, username=user, password=password)
        print(f"  Connected to {host} using password")
    else:
        # Attempt agent / default keys
        client.connect(hostname=host, username=user)
        print(f"  Connected to {host} using SSH agent")

    return client


def pull_weights(host: str, user: str, remote_dir: str,
                 key_path: str = None, password: str = None):
    """Download model weights from server to local models/ directory."""
    try:
        from scp import SCPClient
    except ImportError:
        print("ERROR: scp not installed. Run: pip install scp")
        sys.exit(1)

    client = get_ssh_client(host, user, key_path, password)
    os.makedirs(os.path.join(LOCAL_ROOT, "models"), exist_ok=True)

    with SCPClient(client.get_transport()) as scp:
        for rel_path in MODEL_FILES:
            remote_path = os.path.join(remote_dir, rel_path).replace("\\", "/")
            local_path  = os.path.join(LOCAL_ROOT, rel_path)
            try:
                print(f"  Downloading {remote_path} -> {local_path}")
                scp.get(remote_path, local_path)
                size_mb = os.path.getsize(local_path) / 1024 / 1024
                print(f"    OK ({size_mb:.1f} MB)")
            except Exception as e:
                print(f"    FAILED: {e}")

    client.close()
    print("\nPull complete. Run Streamlit to verify the model loads.")


def push_weights(host: str, user: str, remote_dir: str,
                 key_path: str = None, password: str = None):
    """Upload local model weights to a remote Streamlit host."""
    try:
        from scp import SCPClient
    except ImportError:
        print("ERROR: scp not installed. Run: pip install scp")
        sys.exit(1)

    client = get_ssh_client(host, user, key_path, password)

    with SCPClient(client.get_transport()) as scp:
        for rel_path in MODEL_FILES:
            local_path  = os.path.join(LOCAL_ROOT, rel_path)
            remote_path = os.path.join(remote_dir, rel_path).replace("\\", "/")
            if not os.path.exists(local_path):
                print(f"  SKIPPED (not found locally): {local_path}")
                continue
            size_mb = os.path.getsize(local_path) / 1024 / 1024
            print(f"  Uploading {local_path} ({size_mb:.1f} MB) -> {remote_path}")
            scp.put(local_path, remote_path)
            print(f"    OK")

    client.close()
    print("\nPush complete.")


def show_status(host: str, user: str, remote_dir: str,
                key_path: str = None, password: str = None):
    """Show which model files exist locally and on the server."""
    print("Local files:")
    for rel_path in MODEL_FILES:
        local_path = os.path.join(LOCAL_ROOT, rel_path)
        if os.path.exists(local_path):
            size_mb = os.path.getsize(local_path) / 1024 / 1024
            print(f"  FOUND  {rel_path} ({size_mb:.1f} MB)")
        else:
            print(f"  MISSING {rel_path}")

    print(f"\nRemote files ({host}):")
    client = get_ssh_client(host, user, key_path, password)
    for rel_path in MODEL_FILES:
        remote_path = os.path.join(remote_dir, rel_path).replace("\\", "/")
        stdin, stdout, stderr = client.exec_command(  # nosec B601
            f"ls -lh {remote_path} 2>/dev/null || echo 'NOT FOUND'"
        )
        result = stdout.read().decode().strip()
        print(f"  {result}")
    client.close()


def main():
    parser = argparse.ArgumentParser(description="PACE model weights deployment tool")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pull",   action="store_true", help="Download weights from server")
    group.add_argument("--push",   action="store_true", help="Upload weights to a remote host")
    group.add_argument("--status", action="store_true", help="Show file status on both ends")

    parser.add_argument("--host",       default=os.getenv("DEPLOY_HOST", ""),
                        help="Server hostname (or set DEPLOY_HOST in .env)")
    parser.add_argument("--user",       default=os.getenv("DEPLOY_USER", ""),
                        help="SSH username (or set DEPLOY_USER in .env)")
    parser.add_argument("--key",        default=os.getenv("DEPLOY_KEY", ""),
                        help="Path to SSH private key (optional)")
    parser.add_argument("--password",   default=os.getenv("DEPLOY_PASSWORD", ""),
                        help="SSH password (optional, key preferred)")
    parser.add_argument("--remote-dir", default=os.getenv("DEPLOY_REMOTE_DIR", "~/PACE"),
                        help="Remote repo root directory (default: ~/PACE)")

    args = parser.parse_args()

    if not args.host or not args.user:
        print("ERROR: --host and --user are required.")
        print("Set DEPLOY_HOST and DEPLOY_USER in your .env file or pass as flags.")
        sys.exit(1)

    key_path = args.key if args.key else None
    password = args.password if args.password else None

    if args.pull:
        pull_weights(args.host, args.user, args.remote_dir, key_path, password)
    elif args.push:
        push_weights(args.host, args.user, args.remote_dir, key_path, password)
    elif args.status:
        show_status(args.host, args.user, args.remote_dir, key_path, password)


if __name__ == "__main__":
    main()
