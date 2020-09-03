import io
import logging
from unittest import mock

import pytest

from mim.util.logs import get_logger


class TestLogs:
    def test_get_logger_only_name_param(self):
        log = get_logger('test_stream')

        handler = log.handlers[0]

        assert isinstance(handler, logging.StreamHandler)
        assert logging.DEBUG == log.level
        assert logging.DEBUG == handler.level

    def test_logger_writes_to_stream(self):
        stream_name = 'test_stream'
        stream_message = 'test_message'

        log = get_logger(stream_name)

        test_stream = io.StringIO()

        # replace the real stream with a stream we have control over
        log.handlers[0].stream = test_stream

        log.debug(stream_message)

        stream_content = test_stream.getvalue()
        test_stream.close()

        assert stream_content is not None
        assert '' != stream_content

        # We are not checkning the entire string due to time dependence
        debug_message = f'{stream_name} - DEBUG - {stream_message}\n'
        assert debug_message == ' - '.join(stream_content.split(' - ')[1:])

    @mock.patch('mim.util.logs.logging.FileHandler._open')
    def test_get_logger_returns_a_logger_which_is_opening_a_file(self,
                                                                 mock_open):
        out_file = 'test_log.log'
        log = get_logger('test_file',
                         log_type='DISK',
                         output_file=out_file)

        handler = log.handlers[0]

        assert isinstance(handler, logging.FileHandler)
        assert logging.DEBUG == log.level
        assert logging.DEBUG == handler.level
        mock_open.assert_called_once()

    def test_log_type_disk_without_output_param_raises_value_error(self):
        with pytest.raises(ValueError):
            get_logger('test_file', log_type='DISK')
