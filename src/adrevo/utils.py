import io
import json
import logging
import os
import re
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)


PARENT_RUN_ARTIFACTS_TO_OMIT = frozenset({
    "job_log.err",
    "job_log.out",
    "results.json",
    "returncode.json",
})

DEFAULT_ZIP_EXCLUDE_DIRS = ("__pycache__",)

_CODE_FENCE_LINE_RE = re.compile(r"^[ \t]*```(?P<info>[^\n`]*)[ \t]*$")
_CODE_FENCE_CLOSE_RE = re.compile(r"^[ \t]*```[ \t]*$")


def zip_dir_to_bytes(
    dir_path: str | Path,
    exclude_dirs: tuple = DEFAULT_ZIP_EXCLUDE_DIRS,
) -> bytes:
    """Compresses a directory into an in-memory zip archive.

    Args:
        dir_path: Root directory to zip.
        exclude_dirs: Relative directory paths to exclude in addition to the
            default directory-name exclusions (currently ("__pycache__",)).
    """
    dir_path = Path(dir_path)
    exclude_names = set(DEFAULT_ZIP_EXCLUDE_DIRS)
    exclude_abs = {str((dir_path / d).resolve()) for d in exclude_dirs}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(dir_path):
            # Skip hidden directories and excluded data directories.
            dirs[:] = [
                d
                for d in dirs
                if not d.startswith(".")
                and d not in exclude_names
                and str(Path(root, d).resolve()) not in exclude_abs
            ]
            for file in files:
                if file.startswith("."):
                    continue
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, dir_path)
                zf.write(file_path, arcname)
    return buffer.getvalue()


def zip_data_dirs_to_bytes(dir_path: str | Path, data_dirs: tuple) -> bytes:
    """Zip only the specified data directories within a project directory.

    Args:
        dir_path: Root project directory.
        data_dirs: Relative directory names to include (e.g. ("valid_instances",)).

    Returns:
        Zip bytes containing only the data directories with paths relative to dir_path.
    """
    dir_path = Path(dir_path)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for data_dir in data_dirs:
            data_path = dir_path / data_dir
            if not data_path.is_dir():
                logger.warning(f"Data directory not found, skipping: {data_path}")
                continue
            for root, dirs, files in os.walk(data_path):
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                for file in files:
                    if file.startswith("."):
                        continue
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, dir_path)
                    zf.write(file_path, arcname)
    return buffer.getvalue()


def extract_file_to_string(zip_bytes: bytes, filename: str) -> str:
    """Extracts a single file from an in-memory zip archive as a UTF-8 string."""
    buffer = io.BytesIO(zip_bytes)
    with zipfile.ZipFile(buffer, "r") as zf:
        return zf.read(filename).decode("utf-8")


def extract_bytes_to_dir(zip_bytes: bytes, extract_to: str | Path) -> None:
    """Extracts an in-memory zip archive to a target directory."""
    buffer = io.BytesIO(zip_bytes)
    with zipfile.ZipFile(buffer, "r") as zf:
        zf.extractall(extract_to)


def extract_parent_bytes_to_dir(zip_bytes: bytes, extract_to: str | Path) -> None:
    """Extract a parent program zip while omitting prior run artifacts."""
    buffer = io.BytesIO(zip_bytes)
    extract_root = Path(extract_to)
    with zipfile.ZipFile(buffer, "r") as zf:
        for member in zf.infolist():
            member_path = Path(member.filename)
            if member_path.name in PARENT_RUN_ARTIFACTS_TO_OMIT:
                continue

            zf.extract(member, extract_root)


def parse_results_from_zip(zip_bytes: bytes) -> dict:
    """Parses returncode, results.json, and logs directly from an in-memory zip archive.

    Returns a flat dict with keys: returncode, correct, error, combined_score,
    any other metrics from results.json, stdout_log, stderr_log.
    """
    loaded_results: dict = {"returncode": None, "correct": False, "error": None}
    buffer = io.BytesIO(zip_bytes)

    with zipfile.ZipFile(buffer, "r") as zf:
        namelist = zf.namelist()

        if "returncode.json" in namelist:
            try:
                returncode_results = json.loads(
                    zf.read("returncode.json").decode("utf-8")
                )
                loaded_results.update(returncode_results)
            except json.JSONDecodeError:
                logger.warning("Could not decode JSON from returncode.json in zip.")

        if "results.json" in namelist:
            try:
                loaded_results.update(json.loads(zf.read("results.json").decode("utf-8")))
            except json.JSONDecodeError:
                logger.warning("Could not decode JSON from results.json in zip.")

        loaded_results["stdout_log"] = (
            zf.read("job_log.out").decode("utf-8") if "job_log.out" in namelist else ""
        )
        loaded_results["stderr_log"] = (
            zf.read("job_log.err").decode("utf-8") if "job_log.err" in namelist else ""
        )

    return loaded_results


def extract_last_code_fence(text: str) -> str:
    """
    Extract the code from the last triple-backtick fenced block in ``text``.

    Supports fences with optional info strings like `````python`````.
    Raises ``ValueError`` when no complete fenced block is present.
    """
    if not text:
        raise ValueError("No text provided.")

    lines = text.splitlines(keepends=True)
    in_fence = False
    fence_start_line = 0
    last_block: str | None = None

    for idx, line in enumerate(lines):
        if not in_fence:
            if _CODE_FENCE_LINE_RE.match(line.rstrip("\r\n")):
                in_fence = True
                fence_start_line = idx + 1
            continue

        if _CODE_FENCE_CLOSE_RE.match(line.rstrip("\r\n")):
            last_block = "".join(lines[fence_start_line:idx])
            in_fence = False

    if last_block is None:
        raise ValueError("No complete triple-backtick code fence found.")

    return last_block

def extract_last_code_fence_for_language(text: str, language: str) -> str:
    """
    Extract the code from the last triple-backtick fenced block in ``text``
    whose info string starts with ``language``.

    Examples of matching fences include `````python````` and
    `````python linenums`````.
    Raises ``ValueError`` when no matching complete fenced block is present.
    """
    if not text:
        raise ValueError("No text provided.")

    language = language.strip()
    if not language:
        raise ValueError("No language provided.")

    lines = text.splitlines(keepends=True)
    in_fence = False
    fence_start_line = 0
    current_matches_language = False
    last_block: str | None = None

    for idx, line in enumerate(lines):
        stripped = line.rstrip("\r\n")
        if not in_fence:
            match = _CODE_FENCE_LINE_RE.match(stripped)
            if match:
                in_fence = True
                fence_start_line = idx + 1
                info = match.group("info").strip()
                current_matches_language = (
                    info == language or info.startswith(f"{language} ")
                )
            continue

        if _CODE_FENCE_CLOSE_RE.match(stripped):
            if current_matches_language:
                last_block = "".join(lines[fence_start_line:idx])
            in_fence = False
            current_matches_language = False

    if last_block is None:
        raise ValueError(
            f"No complete triple-backtick code fence found for language '{language}'."
        )

    return last_block


def extract_file_replacements(
    text: str,
    expected_languages: dict[str, str],
) -> dict[str, str]:
    """Extract complete file replacements from a structured LLM response.

    Each replacement must use the following format, where ``path`` is an
    allowed file path and the fence info string exactly matches the language
    configured for that path::

        ### path/to/file
        ```language
        complete replacement contents
        ```

    Text outside replacement sections is ignored.  A response must contain at
    least one replacement, and each path may appear at most once.
    """
    if not text:
        raise ValueError("No text provided.")
    if not expected_languages:
        raise ValueError("No evolvable files are configured.")

    replacements: dict[str, str] = {}
    lines = text.splitlines(keepends=True)
    index = 0

    while index < len(lines):
        header_match = re.match(r"^[ \t]*###\s+(?P<path>\S(?:.*\S)?)[ \t]*$", lines[index].rstrip("\r\n"))
        if header_match is None:
            index += 1
            continue

        file_path = header_match.group("path")
        if file_path not in expected_languages:
            raise ValueError(f"Unexpected file replacement path: {file_path}")
        if file_path in replacements:
            raise ValueError(f"Duplicate file replacement path: {file_path}")

        index += 1
        if index >= len(lines):
            raise ValueError(f"Missing code fence for replacement: {file_path}")
        opening_fence = _CODE_FENCE_LINE_RE.match(lines[index].rstrip("\r\n"))
        if opening_fence is None:
            raise ValueError(f"Missing code fence for replacement: {file_path}")

        language = opening_fence.group("info").strip()
        expected_language = expected_languages[file_path]
        if language != expected_language:
            raise ValueError(
                f"Replacement for {file_path} must use ```{expected_language}``` "
                f"but used ```{language}```."
            )

        index += 1
        contents: list[str] = []
        while index < len(lines):
            if _CODE_FENCE_CLOSE_RE.match(lines[index].rstrip("\r\n")):
                replacements[file_path] = "".join(contents)
                index += 1
                break
            contents.append(lines[index])
            index += 1
        else:
            raise ValueError(f"Unclosed code fence for replacement: {file_path}")

    if not replacements:
        raise ValueError("No structured file replacements found.")
    return replacements
