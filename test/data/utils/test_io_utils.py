import pytest

from topobenchmark.data.utils.io_utils import *


def test_get_file_id_from_url():
    url_1 = "https://docs.google.com/file/d/SOME-FILE-ID-1"
    url_2 = "https://docs.google.com/?id=SOME-FILE-ID-2"
    url_3 = "https://docs.google.com/?arg1=9&id=SOME-FILE-ID-3"
    url_4 = "https://docs.google.com/file/d/idSOME-TRICKY-FILE"
    url_wrong = "https://docs.google.com/?arg1=9&idSOME-FILE-ID"

    assert get_file_id_from_url(url_1) == "SOME-FILE-ID-1"
    assert get_file_id_from_url(url_2) == "SOME-FILE-ID-2"
    assert get_file_id_from_url(url_3) == "SOME-FILE-ID-3"
    assert get_file_id_from_url(url_4) == "idSOME-TRICKY-FILE"

    with pytest.raises(ValueError):
        get_file_id_from_url(url_wrong)
