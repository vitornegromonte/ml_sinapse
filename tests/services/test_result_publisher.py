from unittest.mock import AsyncMock

from services.result_publisher import ResultPublisher


class TestResultPublisher:
    async def test_publish_calls_xadd(self, mock_redis):
        """Arrange: a publisher for ocr:tasks with a maxlen of 100.
        Act: publish a document result.
        Assert: xadd writes to ocr:tasks:results with the document id, text, its
        length as a string, and the maxlen.
        """
        pub = ResultPublisher(mock_redis, "ocr:tasks", maxlen=100)
        await pub.publish({"document_id": "d1"}, "texto extraído")

        mock_redis.xadd.assert_awaited_once()
        args, kwargs = mock_redis.xadd.call_args

        stream = args[0]
        result = args[1]

        assert stream == "ocr:tasks:results"
        assert result["document_id"] == "d1"
        assert result["text"] == "texto extraído"
        assert result["text_length"] == "14"
        assert kwargs.get("maxlen") == 100

    async def test_publish_with_custom_data(self, mock_redis):
        """Arrange: a publisher and a payload with an extra field.
        Act: publish it.
        Assert: document_id is extracted and the text and length are written.
        """
        pub = ResultPublisher(mock_redis, "other:stream", maxlen=50)
        await pub.publish({"document_id": "d2", "extra": "info"}, "outro texto")

        args = mock_redis.xadd.call_args[0]
        result = args[1]

        assert result["document_id"] == "d2"
        assert result["text"] == "outro texto"
        assert result["text_length"] == "11"

    async def test_warns_when_results_stream_is_at_cap(self, mock_redis, caplog):
        """Arrange: a results stream already at its maxlen.
        Act: publish one more result.
        Assert: a warning says the oldest result was dropped — silence here was
        the blind spot of the 1,000 cap.
        """
        import logging

        mock_redis.xlen = AsyncMock(return_value=100)
        pub = ResultPublisher(mock_redis, "ocr:tasks", maxlen=100)

        with caplog.at_level(logging.WARNING):
            await pub.publish({"document_id": "d1"}, "texto")

        assert "cap" in caplog.text

    async def test_no_warning_below_cap(self, mock_redis, caplog):
        """Arrange: a results stream one below its maxlen.
        Act: publish one more result.
        Assert: no warning is logged.
        """
        import logging

        mock_redis.xlen = AsyncMock(return_value=99)
        pub = ResultPublisher(mock_redis, "ocr:tasks", maxlen=100)

        with caplog.at_level(logging.WARNING):
            await pub.publish({"document_id": "d1"}, "texto")

        assert caplog.text == ""
