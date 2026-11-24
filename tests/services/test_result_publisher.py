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


class TestResultPublisherLineage:
    async def test_lineage_fields_are_added_to_every_message(self, mock_redis):
        """Arrange: a publisher built with lineage fields.
        Act: publish two results.
        Assert: both messages carry the manifest_id and model next to the usual fields.
        """
        pub = ResultPublisher(
            mock_redis, "ocr:tasks", maxlen=100,
            lineage={"manifest_id": "abc123def456", "model": "org/m@1234567890ab"},
        )

        await pub.publish({"document_id": "d1"}, "um")
        await pub.publish({"document_id": "d2"}, "dois")

        for call in mock_redis.xadd.call_args_list:
            fields = call.args[1]
            assert fields["manifest_id"] == "abc123def456"
            assert fields["model"] == "org/m@1234567890ab"
            assert {"document_id", "text", "text_length"} <= set(fields)

    async def test_without_lineage_the_message_is_unchanged(self, mock_redis):
        """Arrange: a publisher with no lineage.
        Act: publish a result.
        Assert: only the three original fields are written.
        """
        pub = ResultPublisher(mock_redis, "ocr:tasks", maxlen=100)

        await pub.publish({"document_id": "d1"}, "um")

        assert set(mock_redis.xadd.call_args.args[1]) == {"document_id", "text", "text_length"}


class TestResultPublisherOnLength:
    async def test_the_callback_gets_the_stream_and_its_length(self, mock_redis):
        """Arrange: a publisher with an on_length callback and a stream of 12 entries.
        Act: publish a result.
        Assert: the callback was called once with the results stream name and 12.
        """
        from unittest.mock import MagicMock

        mock_redis.xlen.return_value = 12
        callback = MagicMock()
        pub = ResultPublisher(mock_redis, "ocr:tasks", maxlen=100, on_length=callback)

        await pub.publish({"document_id": "d1"}, "texto")

        callback.assert_called_once_with("ocr:tasks:results", 12)

    async def test_a_failing_callback_does_not_fail_the_publish(self, mock_redis):
        """Arrange: a callback that raises.
        Act: publish a result.
        Assert: the message was still written, and no exception escaped.
        """
        def boom(stream, length):
            raise RuntimeError("metrics broken")

        pub = ResultPublisher(mock_redis, "ocr:tasks", maxlen=100, on_length=boom)

        await pub.publish({"document_id": "d1"}, "texto")

        mock_redis.xadd.assert_awaited_once()

    async def test_no_callback_and_no_maxlen_means_no_length_call(self, mock_redis):
        """Arrange: a publisher with no maxlen.
        Act: publish.
        Assert: XLEN is not called, so nothing is added to the hot path.
        """
        pub = ResultPublisher(mock_redis, "ocr:tasks", maxlen=0)

        await pub.publish({"document_id": "d1"}, "texto")

        mock_redis.xlen.assert_not_awaited()
