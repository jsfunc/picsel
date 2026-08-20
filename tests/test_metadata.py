from pathlib import Path

from PIL import Image

from tamis.metadata import extract_metadata


def _sections_dict(sections):
    return {title: dict(rows) for title, rows in sections}


def test_extract_metadata_basic_image_with_no_exif(tmp_path):
    path = tmp_path / "plain.jpg"
    Image.new("RGB", (5, 4), (0, 0, 0)).save(path)

    sections = _sections_dict(extract_metadata(path))

    assert sections["File"]["Filename"] == "plain.jpg"
    assert sections["Image"]["Dimensions"] == "5 x 4 px"
    assert sections["Image"]["Format"] == "JPEG"
    assert "EXIF" not in sections
    assert "GPS" not in sections


def test_extract_metadata_reads_camera_and_datetime_from_sub_ifd(tmp_path):
    path = tmp_path / "photo.jpg"
    exif = Image.Exif()
    exif[0x010F] = "TestCam"  # Make, IFD0
    exif[0x0110] = "Model X"  # Model, IFD0
    sub = exif.get_ifd(0x8769)
    sub[0x9003] = "2021:07:04 08:15:00"  # DateTimeOriginal, Exif sub-IFD
    Image.new("RGB", (10, 10)).save(path, exif=exif.tobytes())

    sections = _sections_dict(extract_metadata(path))

    assert sections["EXIF"]["Make"] == "TestCam"
    assert sections["EXIF"]["Model"] == "Model X"
    assert sections["EXIF"]["DateTimeOriginal"] == "2021:07:04 08:15:00"


def test_extract_metadata_gps_decimal_conversion(tmp_path):
    path = tmp_path / "geo.jpg"
    exif = Image.Exif()
    gps = exif.get_ifd(0x8825)
    gps[1] = "N"  # GPSLatitudeRef
    gps[2] = (48.0, 51.0, 29.0)  # GPSLatitude (deg, min, sec)
    gps[3] = "E"  # GPSLongitudeRef
    gps[4] = (2.0, 17.0, 40.0)  # GPSLongitude
    Image.new("RGB", (10, 10)).save(path, exif=exif.tobytes())

    sections = _sections_dict(extract_metadata(path))

    lat_str, lon_str = sections["GPS"]["Location"].split(", ")
    assert abs(float(lat_str) - 48.858056) < 1e-4
    assert abs(float(lon_str) - 2.294444) < 1e-4


def test_extract_metadata_gps_south_west_are_negative(tmp_path):
    path = tmp_path / "geo_sw.jpg"
    exif = Image.Exif()
    gps = exif.get_ifd(0x8825)
    gps[1] = "S"
    gps[2] = (33.0, 51.0, 0.0)
    gps[3] = "W"
    gps[4] = (18.0, 25.0, 0.0)
    Image.new("RGB", (10, 10)).save(path, exif=exif.tobytes())

    sections = _sections_dict(extract_metadata(path))
    lat_str, lon_str = sections["GPS"]["Location"].split(", ")
    assert float(lat_str) < 0
    assert float(lon_str) < 0


def test_extract_metadata_nonexistent_file_reports_error_without_raising(tmp_path):
    missing = tmp_path / "does_not_exist.jpg"
    sections = _sections_dict(extract_metadata(missing))
    assert sections["File"]["Filename"] == "does_not_exist.jpg"
    assert "Error" in sections["File"]
