# ============================================================
# GOOGLE DRIVE ROOT CLEANUP
# Designed for millions of files directly under My Drive
#
# SAFETY:
#   - Protects "Colab Notebooks" folder
#   - Protects the Apps Script project ID
#   - Only targets DIRECT children of My Drive root
#   - Moves files to TRASH, does NOT permanently delete
#   - DRY_RUN=True by default
#   - Requires explicit confirmation before real deletion
#
# If Colab disconnects, simply run this cell again.
# ============================================================


# ============================================================
# 1. INSTALL DEPENDENCIES
# ============================================================

!pip -q install --upgrade \
    google-api-python-client \
    google-auth \
    google-auth-httplib2 \
    google-auth-oauthlib \
    requests


# ============================================================
# 2. IMPORTS
# ============================================================

import os
import re
import json
import time
import uuid
import random
import requests

from datetime import datetime

from google.colab import auth
from google.auth import default
from google.auth.transport.requests import Request
from googleapiclient.discovery import build


# ============================================================
# 3. CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# IMPORTANT:
#
# First run with DRY_RUN = True.
#
# After you inspect the result and are satisfied,
# change it to:
#
#     DRY_RUN = False
#
# ------------------------------------------------------------

DRY_RUN = True


# Number of files returned by each Drive list request.
#
# Google Drive may return fewer than this even when more
# results exist, so we always use nextPageToken.
LIST_PAGE_SIZE = 1000


# Google Drive HTTP batch maximum is 100 requests.
BATCH_SIZE = 100


# Print progress every N batches.
PROGRESS_EVERY_BATCHES = 10


# Maximum number of retries for transient errors.
MAX_RETRIES = 8


# Delay between retries.
INITIAL_BACKOFF = 2.0


# The folder that contains all your Colab notebooks.
#
# We will FIND this folder automatically.
PROTECTED_COLAB_FOLDER_NAME = "Colab Notebooks"


# Your Apps Script project ID.
#
# This is the Apps Script ID you previously provided.
#
# It will be protected if it exists in Drive.
APPS_SCRIPT_ID = (
    "1djBOTip6bIy7HaQwN_AJZmxU6Oc10CrehfoVAHC2VHapqBoH-NnX3rPD"
)


# Files/folders can also be manually added here if desired.
#
# Example:
#
# EXTRA_PROTECTED_IDS = {
#     "some_drive_file_id",
# }
#
EXTRA_PROTECTED_IDS = set()


# ============================================================
# 4. AUTHENTICATE
# ============================================================

print("=" * 70)
print("AUTHENTICATING")
print("=" * 70)

auth.authenticate_user()

credentials, project = default(
    scopes=["https://www.googleapis.com/auth/drive"]
)

drive = build(
    "drive",
    "v3",
    credentials=credentials,
    cache_discovery=False,
)

print("Authentication successful.")


# ============================================================
# 5. AUTH HELPER
# ============================================================

def ensure_credentials():
    """
    Refresh OAuth credentials if necessary.
    """

    global credentials

    if not credentials.valid:

        if credentials.expired and credentials.refresh_token:
            print("Refreshing Google credentials...")
            credentials.refresh(Request())

        else:
            raise RuntimeError(
                "Google credentials are invalid and cannot be refreshed."
            )


# ============================================================
# 6. GENERIC RETRY HELPER
# ============================================================

RETRYABLE_STATUS_CODES = {
    429,
    500,
    502,
    503,
    504,
}


def sleep_backoff(attempt):

    delay = INITIAL_BACKOFF * (2 ** attempt)

    # Add a little jitter so repeated requests don't synchronize.
    delay += random.uniform(0, 1)

    delay = min(delay, 60)

    print(f"  Retrying in {delay:.1f} seconds...")

    time.sleep(delay)


# ============================================================
# 7. FIND COLAB NOTEBOOKS FOLDER
# ============================================================

def find_colab_notebooks_folder():

    print()
    print("=" * 70)
    print("LOCATING PROTECTED COLAB FOLDER")
    print("=" * 70)

    query = (
        "'root' in parents "
        "and name = 'Colab Notebooks' "
        "and mimeType = "
        "'application/vnd.google-apps.folder' "
        "and trashed = false"
    )

    response = drive.files().list(
        q=query,
        spaces="drive",
        pageSize=100,
        fields="files(id,name,mimeType,parents)",
    ).execute()

    folders = response.get("files", [])

    if len(folders) == 0:

        raise RuntimeError(
            "\n"
            "SAFETY ABORT\n"
            "--------------------------------------------------\n"
            "Could not find the 'Colab Notebooks' folder directly\n"
            "under My Drive.\n"
            "\n"
            "The script REFUSES to delete anything.\n"
            "--------------------------------------------------"
        )

    if len(folders) > 1:

        print("WARNING: Multiple 'Colab Notebooks' folders found:")

        for folder in folders:
            print(
                f"  {folder['name']} "
                f"ID={folder['id']}"
            )

        raise RuntimeError(
            "\n"
            "SAFETY ABORT\n"
            "Multiple 'Colab Notebooks' folders were found.\n"
            "The script refuses to continue."
        )

    folder = folders[0]

    print(
        f"Protected folder:\n"
        f"  Name: {folder['name']}\n"
        f"  ID:   {folder['id']}"
    )

    return folder["id"]


# ============================================================
# 8. BUILD PROTECTED ID SET
# ============================================================

COLAB_FOLDER_ID = find_colab_notebooks_folder()

PROTECTED_IDS = set()

# Protect Colab Notebooks folder.
PROTECTED_IDS.add(COLAB_FOLDER_ID)

# Protect Apps Script project.
if APPS_SCRIPT_ID:
    PROTECTED_IDS.add(APPS_SCRIPT_ID)

# Protect anything manually specified.
PROTECTED_IDS.update(EXTRA_PROTECTED_IDS)


print()
print("=" * 70)
print("PROTECTED DRIVE ITEMS")
print("=" * 70)

print(f"Colab Notebooks folder:")
print(f"  {COLAB_FOLDER_ID}")

print()
print("Apps Script project:")
print(f"  {APPS_SCRIPT_ID}")

if EXTRA_PROTECTED_IDS:
    print()
    print("Additional protected IDs:")
    for item_id in EXTRA_PROTECTED_IDS:
        print(f"  {item_id}")

print()
print(f"Total protected IDs: {len(PROTECTED_IDS)}")


# ============================================================
# 9. VERIFY PROTECTED COLAB FOLDER
# ============================================================

def verify_protected_folder():

    print()
    print("=" * 70)
    print("VERIFYING COLAB FOLDER")
    print("=" * 70)

    try:

        folder = drive.files().get(
            fileId=COLAB_FOLDER_ID,
            fields="id,name,mimeType,trashed,parents",
        ).execute()

    except Exception as e:

        raise RuntimeError(
            "Could not verify the Colab Notebooks folder."
        ) from e

    if folder.get("trashed", False):

        raise RuntimeError(
            "SAFETY ABORT: Colab Notebooks folder is already in Trash."
        )

    if folder.get("name") != PROTECTED_COLAB_FOLDER_NAME:

        raise RuntimeError(
            "SAFETY ABORT: Protected folder name does not match."
        )

    if folder.get("mimeType") != "application/vnd.google-apps.folder":

        raise RuntimeError(
            "SAFETY ABORT: Protected item is not a folder."
        )

    print("✓ Colab Notebooks folder verified.")
    print(f"  ID: {folder['id']}")
    print(f"  Name: {folder['name']}")
    print(f"  Parent: {folder.get('parents')}")


verify_protected_folder()


# ============================================================
# 10. VERIFY APPS SCRIPT ID
# ============================================================

def verify_apps_script():

    if not APPS_SCRIPT_ID:
        return

    print()
    print("=" * 70)
    print("VERIFYING APPS SCRIPT")
    print("=" * 70)

    try:

        item = drive.files().get(
            fileId=APPS_SCRIPT_ID,
            fields="id,name,mimeType,trashed",
        ).execute()

        print("✓ Apps Script ID exists in Drive.")
        print(f"  ID:       {item['id']}")
        print(f"  Name:     {item.get('name')}")
        print(f"  MIME type:{item.get('mimeType')}")

    except Exception as e:

        print(
            "WARNING: Apps Script ID could not be verified.\n"
            "It will still remain in the protected-ID set.\n"
        )

        print(e)


verify_apps_script()


# ============================================================
# 11. LIST DIRECT CHILDREN OF MY DRIVE ROOT
# ============================================================

def list_root_items(page_token=None):

    query = (
        "'root' in parents "
        "and trashed = false"
    )

    for attempt in range(MAX_RETRIES):

        try:

            ensure_credentials()

            response = drive.files().list(
                q=query,
                spaces="drive",
                pageSize=LIST_PAGE_SIZE,
                pageToken=page_token,
                fields=(
                    "nextPageToken,"
                    "files(id,name,mimeType,parents)"
                ),
                orderBy=None,
            ).execute()

            return response

        except Exception as e:

            message = str(e)

            retry = any(
                str(code) in message
                for code in RETRYABLE_STATUS_CODES
            )

            if attempt >= MAX_RETRIES - 1:

                raise

            if retry:

                sleep_backoff(attempt)

            else:

                raise


# ============================================================
# 12. COUNT ROOT ITEMS
# ============================================================

def count_root_items():

    print()
    print("=" * 70)
    print("COUNTING ACTIVE ITEMS DIRECTLY UNDER MY DRIVE")
    print("=" * 70)

    total = 0
    files = 0
    folders = 0
    protected = 0

    page_token = None

    start = time.time()

    while True:

        response = list_root_items(page_token)

        items = response.get("files", [])

        for item in items:

            total += 1

            if item["id"] in PROTECTED_IDS:

                protected += 1
                continue

            if item["mimeType"] == (
                "application/vnd.google-apps.folder"
            ):
                folders += 1
            else:
                files += 1

        page_token = response.get("nextPageToken")

        if not page_token:
            break

        if total % 100000 == 0:

            elapsed = time.time() - start

            print(
                f"Scanned: {total:,} "
                f"| elapsed: {elapsed / 60:.1f} min"
            )

    elapsed = time.time() - start

    print()
    print("ROOT INVENTORY")
    print("-" * 70)
    print(f"Active root items:       {total:,}")
    print(f"Files:                   {files:,}")
    print(f"Folders:                 {folders:,}")
    print(f"Protected items:         {protected:,}")
    print(f"Potentially removable:   {files + folders:,}")
    print(f"Time:                    {elapsed / 60:.2f} minutes")
    print("-" * 70)

    return {
        "total": total,
        "files": files,
        "folders": folders,
        "protected": protected,
    }


# ============================================================
# 13. GET FIRST INVENTORY
# ============================================================

inventory = count_root_items()


# ============================================================
# 14. DRY RUN
# ============================================================

if DRY_RUN:

    print()
    print("=" * 70)
    print("DRY RUN COMPLETE")
    print("=" * 70)

    print(
        f"\nThe script found approximately "
        f"{inventory['files'] + inventory['folders']:,} "
        f"root-level items that could be removed."
    )

    print()
    print("NOTHING HAS BEEN DELETED OR TRASHED.")
    print()
    print("The following is protected:")
    print("  ✓ Colab Notebooks folder")
    print("  ✓ Apps Script project")
    print("  ✓ Any IDs in EXTRA_PROTECTED_IDS")
    print()
    print("If this looks correct, change:")
    print()
    print("    DRY_RUN = True")
    print()
    print("to:")
    print()
    print("    DRY_RUN = False")
    print()
    print("Then run the entire cell again.")

    raise SystemExit


# ============================================================
# 15. EXPLICIT SAFETY CONFIRMATION
# ============================================================

print()
print("=" * 70)
print("!!! REAL DELETION MODE !!!")
print("=" * 70)

print()
print(
    "This will MOVE root-level My Drive items to TRASH."
)

print()
print("PROTECTED:")
print(f"  ✓ Colab Notebooks: {COLAB_FOLDER_ID}")
print(f"  ✓ Apps Script:      {APPS_SCRIPT_ID}")

print()
print(
    f"Potentially removable items from the initial scan: "
    f"{inventory['files'] + inventory['folders']:,}"
)

print()
print(
    "This script will NOT permanently delete these items."
)
print(
    "They will be moved to Google Drive Trash."
)

print()
confirmation = input(
    'Type exactly "DELETE ROOT FILES" to continue: '
)

if confirmation != "DELETE ROOT FILES":

    print()
    print("ABORTED. Nothing was changed.")
    raise SystemExit


# ============================================================
# 16. BATCH REQUEST BUILDER
# ============================================================

BATCH_URL = "https://www.googleapis.com/batch/drive/v3"


def build_batch_body(items):

    boundary = (
        "===============drive_cleanup_"
        + uuid.uuid4().hex
        + "=="
    )

    parts = []

    for index, item in enumerate(items, start=1):

        file_id = item["id"]

        # We only change metadata:
        # trashed = true
        body = '{"trashed":true}'

        part = (
            f"--{boundary}\r\n"
            f"Content-Type: application/http\r\n"
            f"Content-ID: <request-{index}>\r\n"
            f"\r\n"
            f"PATCH /drive/v3/files/{file_id}"
            f"?fields=id,trashed HTTP/1.1\r\n"
            f"Content-Type: application/json\r\n"
            f"\r\n"
            f"{body}\r\n"
        )

        parts.append(part)

    parts.append(f"--{boundary}--\r\n")

    body = "".join(parts).encode("utf-8")

    content_type = (
        f"multipart/mixed; boundary={boundary}"
    )

    return body, content_type


# ============================================================
# 17. PARSE BATCH RESPONSE
# ============================================================

def parse_batch_statuses(response_text):

    # Each individual response contains something like:
    #
    # HTTP/1.1 200 OK
    #
    # or
    #
    # HTTP/1.1 403 Forbidden
    #
    statuses = re.findall(
        r"HTTP/\d(?:\.\d)?\s+(\d{3})",
        response_text,
    )

    return [int(x) for x in statuses]


# ============================================================
# 18. SEND ONE BATCH
# ============================================================

def send_batch(items):

    ensure_credentials()

    access_token = credentials.token

    body, content_type = build_batch_body(items)

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": content_type,
    }

    for attempt in range(MAX_RETRIES):

        try:

            response = requests.post(
                BATCH_URL,
                headers=headers,
                data=body,
                timeout=120,
            )

            status = response.status_code

            # Outer request succeeded.
            if status == 200:

                statuses = parse_batch_statuses(
                    response.text
                )

                success = sum(
                    1 for code in statuses
                    if 200 <= code < 300
                )

                failed = len(statuses) - success

                # If Google returned fewer individual responses
                # than requests, count those as failures so that
                # the next fresh scan will retry them.
                if len(statuses) < len(items):

                    failed += (
                        len(items) - len(statuses)
                    )

                return success, failed, statuses

            # Authentication expired.
            if status == 401:

                print("OAuth token expired. Refreshing...")

                credentials.refresh(Request())

                headers["Authorization"] = (
                    f"Bearer {credentials.token}"
                )

                continue

            # Retry transient failures.
            if status in RETRYABLE_STATUS_CODES:

                print(
                    f"Batch HTTP {status}; retry "
                    f"{attempt + 1}/{MAX_RETRIES}"
                )

                sleep_backoff(attempt)
                continue

            # Permanent/unknown outer error.
            print(
                "Batch request failed:"
            )
            print(
                response.text[:2000]
            )

            if attempt >= MAX_RETRIES - 1:

                return 0, len(items), []

            sleep_backoff(attempt)

        except requests.RequestException as e:

            print(
                f"Network error: {e}"
            )

            if attempt >= MAX_RETRIES - 1:

                return 0, len(items), []

            sleep_backoff(attempt)

    return 0, len(items), []


# ============================================================
# 19. PROCESS ONE ROOT PAGE
# ============================================================

def process_root_page(items):

    # --------------------------------------------------------
    # SAFETY FILTER
    # --------------------------------------------------------

    candidates = []

    for item in items:

        file_id = item["id"]

        # NEVER touch protected items.
        if file_id in PROTECTED_IDS:

            print(
                f"🛡️ PROTECTED — skipping: "
                f"{item.get('name')} "
                f"({file_id})"
            )

            continue

        # Extra safety:
        # make absolutely sure the item is directly under root.
        parents = item.get("parents", [])

        if "root" not in parents:

            print(
                f"⚠️ SKIPPING non-root item: "
                f"{item.get('name')}"
            )

            continue

        candidates.append(item)

    if not candidates:

        return 0, 0, 0

    total_success = 0
    total_failed = 0
    batch_count = 0

    for start in range(
        0,
        len(candidates),
        BATCH_SIZE,
    ):

        batch = candidates[
            start:start + BATCH_SIZE
        ]

        success, failed, statuses = send_batch(
            batch
        )

        total_success += success
        total_failed += failed
        batch_count += 1

    return (
        len(candidates),
        total_success,
        total_failed,
    )


# ============================================================
# 20. MAIN CLEANUP LOOP
# ============================================================

def cleanup_drive():

    print()
    print("=" * 70)
    print("STARTING DRIVE CLEANUP")
    print("=" * 70)

    print()
    print("Strategy:")
    print("  1. Scan direct children of My Drive root")
    print("  2. Skip protected items")
    print("  3. Trash up to 100 items per HTTP batch")
    print("  4. Rescan from root")
    print("  5. Continue until no items remain")
    print()
    print("Colab Notebooks is protected.")
    print("Apps Script is protected.")
    print()

    start_time = time.time()

    total_seen = 0
    total_success = 0
    total_failed = 0

    iteration = 0
    batches = 0

    while True:

        iteration += 1

        print()
        print("=" * 70)
        print(
            f"SCAN #{iteration}"
        )
        print("=" * 70)

        # ----------------------------------------------------
        # Fresh root scan.
        #
        # We intentionally do NOT depend on an old page token
        # after deletion. The root set is changing continuously.
        # ----------------------------------------------------

        response = list_root_items()

        items = response.get("files", [])

        print(
            f"Drive returned "
            f"{len(items):,} active root items."
        )

        if not items:

            print()
            print("No more root-level active items found.")
            break

        candidates = [
            item
            for item in items
            if item["id"] not in PROTECTED_IDS
            and "root" in item.get("parents", [])
        ]

        protected_count = (
            len(items) - len(candidates)
        )

        print(
            f"Candidates: {len(candidates):,}"
        )

        print(
            f"Protected/skipped: "
            f"{protected_count:,}"
        )

        if not candidates:

            print()
            print(
                "No deletable root-level items "
                "were found in this scan."
            )

            break

        # ----------------------------------------------------
        # Process candidates.
        # ----------------------------------------------------

        for start in range(
            0,
            len(candidates),
            BATCH_SIZE,
        ):

            batch = candidates[
                start:start + BATCH_SIZE
            ]

            success, failed, statuses = (
                send_batch(batch)
            )

            total_seen += len(batch)
            total_success += success
            total_failed += failed
            batches += 1

            if (
                batches % PROGRESS_EVERY_BATCHES == 0
                or failed > 0
            ):

                elapsed = (
                    time.time() - start_time
                )

                rate = (
                    total_success / elapsed
                    if elapsed > 0
                    else 0
                )

                print(
                    f"[Progress] "
                    f"batches={batches:,} | "
                    f"processed={total_seen:,} | "
                    f"trashed={total_success:,} | "
                    f"failed={total_failed:,} | "
                    f"rate={rate:.1f}/sec"
                )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # We immediately rescan from root.
        #
        # Files successfully moved to Trash no longer satisfy:
        #
        #     trashed = false
        #
        # Therefore they disappear from the next scan.
        #
        # Failed files remain and will be retried.
        # ----------------------------------------------------

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    elapsed = time.time() - start_time

    print()
    print("=" * 70)
    print("CLEANUP FINISHED")
    print("=" * 70)

    print(
        f"Processed:       {total_seen:,}"
    )

    print(
        f"Moved to Trash:  {total_success:,}"
    )

    print(
        f"Failed:          {total_failed:,}"
    )

    print(
        f"HTTP batches:    {batches:,}"
    )

    print(
        f"Elapsed time:    {elapsed / 3600:.2f} hours"
    )

    if elapsed > 0:

        print(
            f"Average rate:    "
            f"{total_success / elapsed:.2f} files/sec"
        )

    print()
    print("Protected:")
    print(
        f"  ✓ Colab Notebooks folder: "
        f"{COLAB_FOLDER_ID}"
    )
    print(
        f"  ✓ Apps Script: "
        f"{APPS_SCRIPT_ID}"
    )

    print()
    print(
        "IMPORTANT: Items were moved to Google Drive Trash."
    )

    print(
        "They have NOT been permanently deleted."
    )

    print("=" * 70)


# ============================================================
# 21. RUN
# ============================================================

cleanup_drive()
