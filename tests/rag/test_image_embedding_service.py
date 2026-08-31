"""M1 图片向量批处理基础能力测试。"""

from pathlib import Path

from src.rag.service import image_embedding_service as service


def test_localhost_media_url_is_mapped_to_public_media_host(monkeypatch):
    monkeypatch.setenv("RAG_IMAGE_MEDIA_BASE_URL", "https://ai.smartoptiks.cn")
    assert service._resolve_image_url(
        "http://127.0.0.1:8001/zhangbi_feedback/media/a.webp"
    ) == "https://ai.smartoptiks.cn/zhangbi_feedback/media/a.webp"


def test_non_local_url_is_unchanged():
    url = "https://cdn.example.com/a.webp?x=1"
    assert service._resolve_image_url(url) == url


def test_empty_image_source_is_rejected_without_file():
    try:
        service._download_image_for_embedding("")
    except ValueError:
        pass
    else:
        raise AssertionError("empty image source should be rejected")


def test_existing_local_file_is_not_temporary(tmp_path):
    image = Path(tmp_path) / "sample.webp"
    image.write_bytes(b"not-a-real-image")
    path, temporary = service._download_image_for_embedding(str(image))
    assert path == str(image.resolve())
    assert temporary is False
def test_remote_image_is_downloaded_to_controlled_cache_and_can_be_cleaned(monkeypatch, tmp_path):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b"image-bytes"

    monkeypatch.setenv("RAG_IMAGE_MEDIA_BASE_URL", "https://ai.smartoptiks.cn")
    monkeypatch.setattr(service, "IMAGE_CACHE_ROOT", Path(tmp_path))
    monkeypatch.setattr(service, "urlopen", lambda _request, timeout: Response())
    path, temporary = service._download_image_for_embedding(
        "http://127.0.0.1:8001/zhangbi_feedback/media/a.webp"
    )
    assert temporary is True
    assert Path(path).read_bytes() == b"image-bytes"
    Path(path).unlink()
    assert not Path(path).exists()
