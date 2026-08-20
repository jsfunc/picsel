"""Reads file/image/EXIF/GPS metadata for display. No Qt dependency."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PIL import ExifTags, Image

EXIF_IFD_TAG = 0x8769  # "Exif IFD Pointer": where cameras store most shooting details
GPS_IFD_TAG = 0x8825

_GPS_LAT_REF_TAG = 1
_GPS_LAT_TAG = 2
_GPS_LON_REF_TAG = 3
_GPS_LON_TAG = 4

MetadataSection = tuple[str, list[tuple[str, str]]]


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _format_value(value: object) -> str:
    if isinstance(value, bytes):
        decoded = value.decode("utf-8", errors="replace").strip("\x00").strip()
        return decoded if decoded else value.hex()
    if isinstance(value, str):
        return value.strip("\x00").strip()
    if isinstance(value, tuple):
        return ", ".join(_format_value(v) for v in value)
    if hasattr(value, "numerator") and hasattr(value, "denominator"):
        # PIL's IFDRational (used for exposure time, f-number, GPS components, ...)
        try:
            return "0" if value.denominator == 0 else f"{float(value):g}"
        except (TypeError, ZeroDivisionError):
            return str(value)
    return str(value)


def _gps_decimal(coord: object, ref: object) -> float | None:
    try:
        degrees, minutes, seconds = (float(v) for v in coord)  # type: ignore[misc]
    except (TypeError, ValueError):
        return None
    decimal = degrees + minutes / 60 + seconds / 3600
    if ref in ("S", "W"):
        decimal = -decimal
    return decimal


def _exif_rows(ifd: dict, gps_tags: bool = False) -> list[tuple[str, str]]:
    names = ExifTags.GPSTAGS if gps_tags else ExifTags.TAGS
    rows = []
    for tag_id, value in sorted(ifd.items()):
        label = names.get(tag_id, f"Tag 0x{tag_id:04X}")
        rows.append((label, _format_value(value)))
    return rows


def extract_metadata(path: Path) -> list[MetadataSection]:
    """Return all available metadata for `path`, grouped into display sections."""
    path = Path(path)
    sections: list[MetadataSection] = []

    try:
        stat = path.stat()
        sections.append((
            "File",
            [
                ("Filename", path.name),
                ("Folder", str(path.parent)),
                ("Size", _human_size(stat.st_size)),
                ("Modified", datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")),
            ],
        ))
    except OSError as exc:
        return [("File", [("Filename", path.name), ("Error", str(exc))])]

    try:
        with Image.open(path) as img:
            image_rows = [
                ("Format", img.format or "Unknown"),
                ("Dimensions", f"{img.width} x {img.height} px"),
                ("Color Mode", img.mode),
            ]
            sections.append(("Image", image_rows))

            exif = img.getexif()
            if not exif:
                return sections

            exif_rows = [
                (label, val) for label, val in _exif_rows(exif) if label not in ("ExifOffset", "GPSInfo")
            ]

            sub_ifd: dict = {}
            try:
                sub_ifd = exif.get_ifd(EXIF_IFD_TAG)
            except (KeyError, ValueError):
                pass
            exif_rows.extend(_exif_rows(sub_ifd))
            if exif_rows:
                sections.append(("EXIF", exif_rows))

            gps_ifd: dict = {}
            try:
                gps_ifd = exif.get_ifd(GPS_IFD_TAG)
            except (KeyError, ValueError):
                pass
            if gps_ifd:
                gps_rows = []
                lat = _gps_decimal(gps_ifd.get(_GPS_LAT_TAG), gps_ifd.get(_GPS_LAT_REF_TAG))
                lon = _gps_decimal(gps_ifd.get(_GPS_LON_TAG), gps_ifd.get(_GPS_LON_REF_TAG))
                if lat is not None and lon is not None:
                    gps_rows.append(("Location", f"{lat:.6f}, {lon:.6f}"))
                gps_rows.extend(_exif_rows(gps_ifd, gps_tags=True))
                sections.append(("GPS", gps_rows))
    except Exception as exc:  # noqa: BLE001 - metadata display must never crash the app
        sections.append(("Error", [("Could not read image metadata", str(exc))]))

    return sections
