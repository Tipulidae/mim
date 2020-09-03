import logging


def get_logger(name, log_type='CONSOLE', output_file=None):
    """
    Logging wrapping function based on the standard library logging. With this
    you don't have to set some standard parameters each time you use logging.

    :param name: The name of the logger is reflected in the resulting log file
    or output stream.
    :param log_type: If 'DISK', the log will be saved to the file defined in
    output_file. Default is 'CONSOLE'.
    :param output_file: Path to the file to be written to if the log type is
    'DISK'.
    :return: A Python Logger Object
    """
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)

    if log_type == 'DISK':
        if output_file is None:
            raise ValueError('Output file not set for DISK log type.')
        h = logging.FileHandler(output_file)
    else:
        h = logging.StreamHandler()

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - "
                                  "%(message)s", "%Y-%m-%d %H:%M:%S")
    h.setLevel(logging.DEBUG)
    h.setFormatter(formatter)

    if not log.handlers:
        log.addHandler(h)

    return log
