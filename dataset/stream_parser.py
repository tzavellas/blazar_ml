import logging.config
import re


def find_overflow(stream):
    '''
    Searches for the string "overflow!!!". Returns true if it finds it else false.
        Parameters:
            stream (str):                   Program stdout stream.
    '''
    # search stream for overflow
    overflow = stream.find('overflow!!!')
    return overflow != -1


def find_ifail(stream):
    '''
    Searches for the string "IFAIL". Returns true if it finds it else false.
        Parameters:
            stream (str):                   Program stdout stream.
    '''
    integration_pattern = '.*IFAIL\s=\s*2'          # IFAIL regex
    match = re.search(integration_pattern, stream)
    return match is not None


def find_elapsed_time(stream):
    '''
    Searches for the string "Elapsed CPU time". Returns true if it finds it else false.
        Parameters:
            stream (str):                   Program stdout stream.
    '''
    logger = logging.getLogger(__name__)

    elapsed_time = .0
    pattern = 'Elapsed CPU time\D+(\d+\.\d+)'       # Elapsed CPU regex
    match = re.search(pattern, stream)
    if match:
        elapsed_time = float(match.group(1))
        logger.info('Elapsed CPU time {}'.format(elapsed_time))
    return elapsed_time, match is not None


def parse_stream(stream):
    '''
    Parses a stream and detects unsuccessful execution and extracts execution time.S
        Parameters:
            stream (str):                   Program stdout stream.
    '''
    logger = logging.getLogger(__name__)

    overflow_found = find_overflow(stream)
    logger.debug('overflow found {}'.format(overflow_found))

    ifail_found = find_ifail(stream)
    logger.debug('ifail found {}'.format(ifail_found))

    success = (not overflow_found) and (not ifail_found)
    logger.info('Run was successful {}'.format(success))

    elapsed_time = .0
    if success:
        elapsed_time, found = find_elapsed_time(stream)
        if not found:
            logger.error('Elapsed CPU time not found')
    else:
        logger.warning(
            'Execution was not successful. Elapsed CPU time will be set to 0.')

    return success, elapsed_time
