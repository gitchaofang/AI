# ============================================================
# GOOGLE DRIVE ROOT CLEANUP
# ============================================================
#
# PURPOSE
# -------
# Move files/folders DIRECTLY under My Drive root to Trash.
#
# SAFETY
# ------
# 1. "Colab Notebooks" is automatically protected.
# 2. Your Apps Script project is protected.
# 3. ONLY direct children of My Drive root are targeted.
# 4. Files inside "Colab Notebooks" are NOT targeted.
# 5. Items are MOVED TO TRASH, not permanently deleted.
# 6. DRY_RUN=True by default.
# 7. Real deletion requires explicit confirmation.
# 8. After every batch, Drive is rescanned.
# 9. Failed items remain and can be retried.
# 10. The script is safe to stop/restart.
#
# IMPORTANT
# ---------
# FIRST RUN:
#
#     DRY_RUN = True
#
# Verify the inventory.
#
# ONLY AFTER VERIFYING:
#
#     DRY_RUN = False
#
# Then run the entire cell again.
#
# ============================================================


# ============================================================
# 1. INSTALL REQUIRED PACKAGES
# ============================================================

!pip -q install --upgrade \
    google-api-python-client \
    google-auth \
    google-auth-httplib2 \
    google-auth-oauthlib


# ============================================================
# 2. IMPORTS
# ============================================================

import time
import random
import traceback

from google.colab import auth

import google.auth
from google.auth.transport.requests import Request

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import BatchHttpRequest


# ============================================================
# 3. CONFIGURATION
# ============================================================


# ------------------------------------------------------------
# SAFETY SWITCH
# ------------------------------------------------------------
#
# FIRST RUN:
#
#     DRY_RUN = True
#
# Nothing is changed.
#
# After verifying the inventory:
#
#     DRY_RUN = False
#
# ------------------------------------------------------------

DRY_RUN = True


# ------------------------------------------------------------
# DRIVE LIST PAGE SIZE
# ------------------------------------------------------------
#
# Drive API allows up to 1000 files per list request.
#
LIST_PAGE_SIZE = 1000


# ------------------------------------------------------------
# BATCH SIZE
# ------------------------------------------------------------
#
# Google Drive API batching supports up to 100 requests
# per batch.
#
BATCH_SIZE = 100


# ------------------------------------------------------------
# RETRY CONFIGURATION
# ------------------------------------------------------------

MAX_RETRIES = 8

INITIAL_BACKOFF_SECONDS = 2.0

MAX_BACKOFF_SECONDS = 60.0


# ------------------------------------------------------------
# PROTECTED COLAB FOLDER
# ------------------------------------------------------------

PROTECTED_COLAB_FOLDER_NAME = "Colab Notebooks"


# ------------------------------------------------------------
# PROTECTED APPS SCRIPT
# ------------------------------------------------------------
#
# Your Apps Script project ID.
#
# This ID is always protected.
#
APPS_SCRIPT_ID = (
    "1djBOTip6bIy7HaQwN_AJZmxU6Oc10CrehfoVAHC2VHapqBoH-NnX3rPD"
)


# ------------------------------------------------------------
# ADDITIONAL PROTECTED IDS
# ------------------------------------------------------------
#
# Add other Drive IDs here if desired.
#
# Example:
#
# EXTRA_PROTECTED_IDS = {
#     "1abc...",
#     "1xyz...",
# }
#
# ------------------------------------------------------------

EXTRA_PROTECTED_IDS = set()


# ============================================================
# 4. AUTHENTICATION
# ============================================================

print()
print("=" * 70)
print("AUTHENTICATING WITH GOOGLE DRIVE")
print("=" * 70)

auth.authenticate_user()

credentials, project = google.auth.default(
    scopes=[
        "https://www.googleapis.com/auth/drive"
    ]
)

drive = build(
    "drive",
    "v3",
    credentials=credentials,
    cache_discovery=False,
)

print()
print("✓ Google authentication successful.")


# ============================================================
# 5. ENSURE CREDENTIALS ARE VALID
# ============================================================
#
# IMPORTANT:
#
# Do NOT use:
#
#     credentials.refresh_token
#
# Colab credentials do not necessarily expose that attribute.
#
# Instead use credentials.refresh(Request()).
#
# ============================================================

def ensure_credentials():

    global credentials

    # Access token is still valid.
    if credentials.valid:
        return

    try:

        print(
            "Refreshing Google credentials..."
        )

        credentials.refresh(
            Request()
        )

        print(
            "✓ Credentials refreshed."
        )

    except Exception as e:

        print()
        print("=" * 70)
        print("GOOGLE CREDENTIAL REFRESH FAILED")
        print("=" * 70)

        print()
        print(
            "Please authenticate again by running:"
        )

        print()
        print(
            "    auth.authenticate_user()"
        )

        print()
        print(
            "Then rerun this cell."
        )

        raise RuntimeError(
            "Google credentials could not be refreshed."
        ) from e


# ============================================================
# 6. FIND COLAB NOTEBOOKS FOLDER
# ============================================================
#
# We specifically look for:
#
#     My Drive / Colab Notebooks
#
# The folder itself is protected.
#
# Everything underneath it is naturally protected because
# this script ONLY targets direct children of My Drive root.
#
# ============================================================

def find_colab_notebooks_folder():

    print()
    print("=" * 70)
    print("FINDING PROTECTED COLAB NOTEBOOKS FOLDER")
    print("=" * 70)

    query = (
        "'root' in parents "
        "and name = 'Colab Notebooks' "
        "and mimeType = "
        "'application/vnd.google-apps.folder' "
        "and trashed = false"
    )

    ensure_credentials()

    response = drive.files().list(
        q=query,
        spaces="drive",
        pageSize=100,
        fields=(
            "files("
            "id,"
            "name,"
            "mimeType,"
            "parents"
            ")"
        ),
    ).execute()

    folders = response.get(
        "files",
        []
    )

    # --------------------------------------------------------
    # SAFETY ABORT
    # --------------------------------------------------------

    if len(folders) == 0:

        raise RuntimeError(
            "\n"
            "SAFETY ABORT\n"
            "--------------------------------------------------\n"
            "Could not find 'Colab Notebooks' directly under\n"
            "My Drive root.\n"
            "\n"
            "The script refuses to delete anything.\n"
            "--------------------------------------------------"
        )

    # --------------------------------------------------------
    # Multiple folders are ambiguous.
    # --------------------------------------------------------

    if len(folders) > 1:

        print(
            "Multiple 'Colab Notebooks' folders found:"
        )

        for folder in folders:

            print(
                f"  Name={folder.get('name')} "
                f"ID={folder.get('id')}"
            )

        raise RuntimeError(
            "\n"
            "SAFETY ABORT\n"
            "Multiple 'Colab Notebooks' folders were found.\n"
            "The script refuses to continue."
        )

    folder = folders[0]

    print()
    print(
        "✓ Protected folder found."
    )

    print(
        f"  Name: {folder['name']}"
    )

    print(
        f"  ID:   {folder['id']}"
    )

    return folder["id"]


# ============================================================
# 7. BUILD PROTECTED ID SET
# ============================================================

COLAB_FOLDER_ID = (
    find_colab_notebooks_folder()
)

PROTECTED_IDS = set()


# Protect Colab Notebooks.
PROTECTED_IDS.add(
    COLAB_FOLDER_ID
)


# Protect Apps Script.
if APPS_SCRIPT_ID:

    PROTECTED_IDS.add(
        APPS_SCRIPT_ID
    )


# Protect additional IDs.
PROTECTED_IDS.update(
    EXTRA_PROTECTED_IDS
)


# ============================================================
# 8. PRINT PROTECTED ITEMS
# ============================================================

print()
print("=" * 70)
print("PROTECTED DRIVE ITEMS")
print("=" * 70)

print()
print("✓ Colab Notebooks")
print(
    f"    {COLAB_FOLDER_ID}"
)

print()
print("✓ Apps Script")
print(
    f"    {APPS_SCRIPT_ID}"
)

if EXTRA_PROTECTED_IDS:

    print()
    print("✓ Additional protected IDs:")

    for file_id in sorted(
        EXTRA_PROTECTED_IDS
    ):

        print(
            f"    {file_id}"
        )

print()
print(
    f"Total protected IDs: "
    f"{len(PROTECTED_IDS)}"
)


# ============================================================
# 9. VERIFY COLAB NOTEBOOKS
# ============================================================

def verify_colab_folder():

    print()
    print("=" * 70)
    print("VERIFYING COLAB NOTEBOOKS")
    print("=" * 70)

    ensure_credentials()

    folder = drive.files().get(
        fileId=COLAB_FOLDER_ID,
        fields=(
            "id,"
            "name,"
            "mimeType,"
            "trashed,"
            "parents"
        ),
    ).execute()

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    if folder.get("trashed"):

        raise RuntimeError(
            "SAFETY ABORT: "
            "Colab Notebooks is already in Trash."
        )

    if (
        folder.get("name")
        != PROTECTED_COLAB_FOLDER_NAME
    ):

        raise RuntimeError(
            "SAFETY ABORT: "
            "Protected folder name does not match."
        )

    if (
        folder.get("mimeType")
        != "application/vnd.google-apps.folder"
    ):

        raise RuntimeError(
            "SAFETY ABORT: "
            "Protected item is not a folder."
        )

    print()
    print(
        "✓ Colab Notebooks verified."
    )

    print(
        f"  Name: {folder['name']}"
    )

    print(
        f"  ID:   {folder['id']}"
    )

    print(
        f"  Parent: {folder.get('parents')}"
    )


verify_colab_folder()


# ============================================================
# 10. VERIFY APPS SCRIPT
# ============================================================

def verify_apps_script():

    if not APPS_SCRIPT_ID:

        return

    print()
    print("=" * 70)
    print("VERIFYING APPS SCRIPT")
    print("=" * 70)

    try:

        ensure_credentials()

        item = drive.files().get(
            fileId=APPS_SCRIPT_ID,
            fields=(
                "id,"
                "name,"
                "mimeType,"
                "trashed"
            ),
        ).execute()

        print()
        print(
            "✓ Apps Script ID found."
        )

        print(
            f"  ID:        {item['id']}"
        )

        print(
            f"  Name:      {item.get('name')}"
        )

        print(
            f"  MIME type: {item.get('mimeType')}"
        )

        print(
            f"  Trashed:   {item.get('trashed')}"
        )

    except Exception as e:

        # IMPORTANT:
        #
        # Even if verification fails, the ID remains in
        # PROTECTED_IDS.
        #
        print()
        print(
            "WARNING:"
        )

        print(
            "Could not verify the Apps Script ID."
        )

        print(
            "The ID will nevertheless remain protected."
        )

        print(
            f"Reason: {e}"
        )


verify_apps_script()


# ============================================================
# 11. ROOT QUERY
# ============================================================
#
# THIS IS THE KEY QUERY:
#
#     'root' in parents
#
# means:
#
#     direct children of My Drive root.
#
#     trashed = false
#
# means:
#
#     currently active items.
#
# IMPORTANT:
#
# We DO NOT later test:
#
#     "root" in item["parents"]
#
# because the returned "parents" field contains actual
# folder IDs, not the special "root" alias.
#
# ============================================================

ROOT_QUERY = (
    "'root' in parents "
    "and trashed = false"
)


# ============================================================
# 12. LIST ONE PAGE OF ROOT ITEMS
# ============================================================

def list_root_page(
    page_token=None
):

    for attempt in range(
        MAX_RETRIES
    ):

        try:

            ensure_credentials()

            response = drive.files().list(

                q=ROOT_QUERY,

                spaces="drive",

                pageSize=LIST_PAGE_SIZE,

                pageToken=page_token,

                fields=(
                    "nextPageToken,"
                    "files("
                    "id,"
                    "name,"
                    "mimeType,"
                    "parents"
                    ")"
                ),

            ).execute()

            return response

        except HttpError as e:

            status = (
                e.resp.status
                if e.resp
                else None
            )

            print()
            print(
                f"Drive list error: "
                f"HTTP {status}"
            )

            # ------------------------------------------------
            # Retry transient errors.
            # ------------------------------------------------

            if (
                status in {
                    429,
                    500,
                    502,
                    503,
                    504,
                }
                and attempt
                < MAX_RETRIES - 1
            ):

                sleep_seconds = min(
                    INITIAL_BACKOFF_SECONDS
                    * (2 ** attempt)
                    + random.random(),
                    MAX_BACKOFF_SECONDS,
                )

                print(
                    f"Retrying in "
                    f"{sleep_seconds:.1f} seconds..."
                )

                time.sleep(
                    sleep_seconds
                )

                continue

            raise

        except Exception:

            print(
                "Unexpected error while listing Drive."
            )

            traceback.print_exc()

            raise


# ============================================================
# 13. COUNT ALL ROOT ITEMS
# ============================================================
#
# This function performs a COMPLETE paginated scan.
#
# It is only used for the initial dry-run inventory.
#
# ============================================================

def count_root_items():

    print()
    print("=" * 70)
    print(
        "COUNTING ACTIVE ITEMS DIRECTLY UNDER MY DRIVE ROOT"
    )
    print("=" * 70)

    total = 0
    files = 0
    folders = 0
    protected = 0

    page_token = None

    start_time = time.time()

    while True:

        response = list_root_page(
            page_token
        )

        items = response.get(
            "files",
            []
        )

        for item in items:

            total += 1

            # ------------------------------------------------
            # Protected item
            # ------------------------------------------------

            if item["id"] in PROTECTED_IDS:

                protected += 1

                continue

            # ------------------------------------------------
            # Folder
            # ------------------------------------------------

            if (
                item["mimeType"]
                == "application/vnd.google-apps.folder"
            ):

                folders += 1

            # ------------------------------------------------
            # Regular file
            # ------------------------------------------------

            else:

                files += 1

        page_token = (
            response.get(
                "nextPageToken"
            )
        )

        if not page_token:

            break

        # ----------------------------------------------------
        # Progress report.
        # ----------------------------------------------------

        if (
            total > 0
            and total % 100000 == 0
        ):

            elapsed = (
                time.time()
                - start_time
            )

            print(
                f"Scanned "
                f"{total:,} items..."
            )

            print(
                f"Elapsed: "
                f"{elapsed / 60:.1f} minutes"
            )

    elapsed = (
        time.time()
        - start_time
    )

    removable = (
        files
        + folders
    )

    print()
    print("=" * 70)
    print("ROOT INVENTORY")
    print("=" * 70)

    print(
        f"Active root items:      "
        f"{total:,}"
    )

    print(
        f"Files:                  "
        f"{files:,}"
    )

    print(
        f"Folders:                "
        f"{folders:,}"
    )

    print(
        f"Protected items:        "
        f"{protected:,}"
    )

    print(
        f"Potentially removable:  "
        f"{removable:,}"
    )

    print(
        f"Scan time:              "
        f"{elapsed / 60:.2f} minutes"
    )

    print("=" * 70)

    return {
        "total": total,
        "files": files,
        "folders": folders,
        "protected": protected,
        "removable": removable,
    }


# ============================================================
# 14. INITIAL INVENTORY
# ============================================================

inventory = count_root_items()


# ============================================================
# 15. DRY RUN EXIT
# ============================================================

if DRY_RUN:

    print()
    print("=" * 70)
    print("DRY RUN COMPLETE")
    print("=" * 70)

    print()
    print(
        "NO FILES HAVE BEEN MODIFIED."
    )

    print()
    print(
        f"Potentially removable root items: "
        f"{inventory['removable']:,}"
    )

    print()
    print(
        "Protected:"
    )

    print(
        f"  ✓ Colab Notebooks:"
    )

    print(
        f"      {COLAB_FOLDER_ID}"
    )

    print(
        f"  ✓ Apps Script:"
    )

    print(
        f"      {APPS_SCRIPT_ID}"
    )

    print()
    print(
        "If this inventory is correct, change:"
    )

    print()
    print(
        "    DRY_RUN = True"
    )

    print()
    print(
        "to:"
    )

    print()
    print(
        "    DRY_RUN = False"
    )

    print()
    print(
        "Then rerun the entire cell."
    )

    raise SystemExit


# ============================================================
# 16. REAL CLEANUP CONFIRMATION
# ============================================================

print()
print("=" * 70)
print("!!! REAL CLEANUP MODE !!!")
print("=" * 70)

print()

print(
    "The script will move eligible root-level "
    "items to Google Drive Trash."
)

print()

print(
    "It will NOT permanently delete them."
)

print()

print(
    f"Initial removable count: "
    f"{inventory['removable']:,}"
)

print()

print(
    "Protected:"
)

print(
    f"  ✓ Colab Notebooks"
)

print(
    f"      {COLAB_FOLDER_ID}"
)

print(
    f"  ✓ Apps Script"
)

print(
    f"      {APPS_SCRIPT_ID}"
)

print()

confirmation = input(
    'Type exactly "DELETE ROOT FILES" to continue: '
)


if (
    confirmation
    != "DELETE ROOT FILES"
):

    print()
    print(
        "ABORTED."
    )

    print(
        "Nothing was changed."
    )

    raise SystemExit


# ============================================================
# 17. BATCH STATE
# ============================================================

class BatchState:

    def __init__(self):

        self.success = 0

        self.failed = 0

        self.errors = []

        self.completed = 0


# ============================================================
# 18. BATCH CALLBACK
# ============================================================
#
# IMPORTANT:
#
# This is a NORMAL Python function.
#
# We deliberately do NOT use:
#
#     lambda (
#         request_id,
#         response,
#         exception,
#     )
#
# because that syntax is invalid in Python.
#
# ============================================================

def make_batch_callback(
    state
):

    def callback(
        request_id,
        response,
        exception,
    ):

        state.completed += 1

        if exception is not None:

            state.failed += 1

            state.errors.append(
                (
                    request_id,
                    exception,
                )
            )

            return

        state.success += 1

    return callback


# ============================================================
# 19. TRASH ONE BATCH
# ============================================================

def trash_batch(
    items
):

    ensure_credentials()

    state = BatchState()

    # --------------------------------------------------------
    # Create Drive batch.
    # --------------------------------------------------------

    batch = BatchHttpRequest(
        callback=make_batch_callback(
            state
        )
    )

    # --------------------------------------------------------
    # Add requests.
    # --------------------------------------------------------

    for item in items:

        file_id = item["id"]

        request = drive.files().update(

            fileId=file_id,

            body={
                "trashed": True
            },

            fields=(
                "id,"
                "name,"
                "trashed"
            ),
        )

        batch.add(
            request,
            request_id=file_id,
        )

    # --------------------------------------------------------
    # Execute batch.
    # --------------------------------------------------------
    #
    # Individual request failures are captured by callback().
    #
    # A batch-level failure is retried.
    #
    # Since setting trashed=true is idempotent, retrying a
    # request that may already have succeeded is safe.
    #
    # --------------------------------------------------------

    for attempt in range(
        MAX_RETRIES
    ):

        try:

            ensure_credentials()

            batch.execute()

            return state

        except HttpError as e:

            status = (
                e.resp.status
                if e.resp
                else None
            )

            print()
            print(
                f"Batch execution error: "
                f"HTTP {status}"
            )

            # ------------------------------------------------
            # Retry transient errors.
            # ------------------------------------------------

            if (
                status in {
                    429,
                    500,
                    502,
                    503,
                    504,
                }
                and attempt
                < MAX_RETRIES - 1
            ):

                sleep_seconds = min(
                    INITIAL_BACKOFF_SECONDS
                    * (2 ** attempt)
                    + random.random(),
                    MAX_BACKOFF_SECONDS,
                )

                print(
                    f"Retrying batch in "
                    f"{sleep_seconds:.1f} seconds..."
                )

                time.sleep(
                    sleep_seconds
                )

                continue

            raise

        except Exception as e:

            print()
            print(
                "Unexpected batch execution error:"
            )

            print(
                e
            )

            if (
                attempt
                >= MAX_RETRIES - 1
            ):

                raise

            sleep_seconds = min(
                INITIAL_BACKOFF_SECONDS
                * (2 ** attempt)
                + random.random(),
                MAX_BACKOFF_SECONDS,
            )

            print(
                f"Retrying batch in "
                f"{sleep_seconds:.1f} seconds..."
            )

            time.sleep(
                sleep_seconds
            )

    return state


# ============================================================
# 20. PRINT BATCH ERRORS
# ============================================================

def print_batch_errors(
    state
):

    if not state.errors:

        return

    print()
    print(
        "Individual request errors:"
    )

    # --------------------------------------------------------
    # Print at most 10 to avoid flooding Colab output.
    # --------------------------------------------------------

    for (
        request_id,
        exception,
    ) in state.errors[:10]:

        print(
            f"  ID={request_id}"
        )

        print(
            f"    {exception}"
        )

    if (
        len(state.errors)
        > 10
    ):

        print(
            f"  ... plus "
            f"{len(state.errors) - 10} "
            f"additional errors"
        )


# ============================================================
# 21. LIST FIRST PAGE OF CURRENT ROOT ITEMS
# ============================================================
#
# During deletion we intentionally only need the first page.
#
# Why?
#
# Suppose there are 5,000,000 items:
#
#     get first 100
#          ↓
#     trash 100
#          ↓
#     get first 100 again
#          ↓
#     trash 100
#          ↓
#     ...
#
# Eventually the first page becomes empty.
#
# This avoids holding millions of IDs in memory.
#
# ============================================================

def get_current_candidates():

    response = list_root_page()

    items = response.get(
        "files",
        []
    )

    candidates = []

    protected_count = 0

    for item in items:

        file_id = item["id"]

        # ----------------------------------------------------
        # Protection check
        # ----------------------------------------------------

        if file_id in PROTECTED_IDS:

            protected_count += 1

            continue

        # ----------------------------------------------------
        # The ROOT_QUERY already guarantees:
        #
        #     'root' in parents
        #
        # Therefore every item here is a direct child of
        # My Drive root.
        #
        # DO NOT check:
        #
        #     "root" in item["parents"]
        #
        # ----------------------------------------------------

        candidates.append(
            item
        )

    return (
        candidates,
        protected_count,
    )


# ============================================================
# 22. MAIN CLEANUP LOOP
# ============================================================

def cleanup_drive():

    print()
    print("=" * 70)
    print("STARTING DRIVE CLEANUP")
    print("=" * 70)

    print()
    print(
        "Strategy:"
    )

    print(
        "  1. Scan direct children of My Drive root."
    )

    print(
        "  2. Skip protected items."
    )

    print(
        "  3. Trash at most 100 items."
    )

    print(
        "  4. Perform a fresh scan."
    )

    print(
        "  5. Repeat until no candidates remain."
    )

    print()
    print(
        "Protected Colab folder:"
    )

    print(
        f"  {COLAB_FOLDER_ID}"
    )

    print()
    print(
        "Protected Apps Script:"
    )

    print(
        f"  {APPS_SCRIPT_ID}"
    )

    start_time = time.time()

    total_processed = 0

    total_trashed = 0

    total_failed = 0

    total_batches = 0

    scan_number = 0

    # ========================================================
    # CONTINUOUS LOOP
    # ========================================================

    while True:

        scan_number += 1

        print()
        print("=" * 70)
        print(
            f"SCAN #{scan_number}"
        )
        print("=" * 70)

        # ----------------------------------------------------
        # Fresh scan.
        # ----------------------------------------------------

        (
            candidates,
            protected_count,
        ) = get_current_candidates()

        print()
        print(
            f"Active root items returned: "
            f"{len(candidates) + protected_count:,}"
        )

        print(
            f"Candidates: "
            f"{len(candidates):,}"
        )

        print(
            f"Protected/skipped: "
            f"{protected_count:,}"
        )

        # ----------------------------------------------------
        # Nothing left.
        # ----------------------------------------------------

        if not candidates:

            print()
            print(
                "✓ No deletable root-level "
                "items remain."
            )

            break

        # ----------------------------------------------------
        # Take at most BATCH_SIZE.
        # ----------------------------------------------------

        batch_items = candidates[
            :BATCH_SIZE
        ]

        # ----------------------------------------------------
        # FINAL SAFETY CHECK
        # ----------------------------------------------------

        unsafe_items = [
            item
            for item in batch_items
            if item["id"]
            in PROTECTED_IDS
        ]

        if unsafe_items:

            print()
            print(
                "SAFETY ABORT:"
            )

            print(
                "A protected item entered the deletion batch."
            )

            for item in unsafe_items:

                print(
                    f"  {item.get('name')} "
                    f"[{item.get('id')}]"
                )

            raise RuntimeError(
                "Protected item detected in deletion batch."
            )

        # ----------------------------------------------------
        # Show first few files in batch.
        # ----------------------------------------------------

        print()
        print(
            f"Preparing batch of "
            f"{len(batch_items)} items:"
        )

        for item in batch_items[:10]:

            print(
                f"  {item.get('name')} "
                f"[{item.get('id')}]"
            )

        if len(batch_items) > 10:

            print(
                f"  ... and "
                f"{len(batch_items) - 10} more"
            )

        # ----------------------------------------------------
        # TRASH BATCH
        # ----------------------------------------------------

        state = trash_batch(
            batch_items
        )

        total_batches += 1

        total_processed += (
            len(batch_items)
        )

        total_trashed += (
            state.success
        )

        total_failed += (
            state.failed
        )

        # ----------------------------------------------------
        # REPORT
        # ----------------------------------------------------

        elapsed = (
            time.time()
            - start_time
        )

        rate = (
            total_trashed
            / elapsed
            if elapsed > 0
            else 0
        )

        print()
        print(
            f"Batch #{total_batches:,} complete:"
        )

        print(
            f"  Sent:              "
            f"{len(batch_items):,}"
        )

        print(
            f"  Trashed:           "
            f"{state.success:,}"
        )

        print(
            f"  Failed:            "
            f"{state.failed:,}"
        )

        print(
            f"  Total trashed:     "
            f"{total_trashed:,}"
        )

        print(
            f"  Average rate:      "
            f"{rate:.2f} files/sec"
        )

        if state.failed:

            print_batch_errors(
                state
            )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # We intentionally DO NOT continue using the existing
        # candidate list.
        #
        # We return to the top of the loop and perform a
        # completely fresh Drive API query.
        #
        # Successful files now have:
        #
        #     trashed = true
        #
        # and therefore disappear from ROOT_QUERY.
        #
        # Failed files remain and will be seen again.
        #
        # ----------------------------------------------------

    # ========================================================
    # FINAL REPORT
    # ========================================================

    elapsed = (
        time.time()
        - start_time
    )

    print()
    print("=" * 70)
    print("CLEANUP FINISHED")
    print("=" * 70)

    print()
    print(
        f"Scans:             "
        f"{scan_number:,}"
    )

    print(
        f"Batches:           "
        f"{total_batches:,}"
    )

    print(
        f"Processed:         "
        f"{total_processed:,}"
    )

    print(
        f"Moved to Trash:    "
        f"{total_trashed:,}"
    )

    print(
        f"Failed requests:   "
        f"{total_failed:,}"
    )

    print(
        f"Elapsed time:      "
        f"{elapsed / 3600:.2f} hours"
    )

    if elapsed > 0:

        print(
            f"Average rate:      "
            f"{total_trashed / elapsed:.2f} files/sec"
        )

    print()
    print(
        "PROTECTED ITEMS:"
    )

    print(
        "  ✓ Colab Notebooks:"
    )

    print(
        f"      {COLAB_FOLDER_ID}"
    )

    print(
        "  ✓ Apps Script:"
    )

    print(
        f"      {APPS_SCRIPT_ID}"
    )

    print()
    print(
        "The items were moved to Google Drive Trash."
    )

    print(
        "They were NOT permanently deleted."
    )

    print()
    print("=" * 70)


# ============================================================
# 23. RUN CLEANUP
# ============================================================

cleanup_drive()
