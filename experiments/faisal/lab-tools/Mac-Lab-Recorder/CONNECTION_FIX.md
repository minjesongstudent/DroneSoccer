Recorder connection fix - 6 October 2026

Three supplied session exports contained no telemetry or video. The first timed
out; the next two received binary packets on the command socket and stopped
before requesting streamon. Their decoded payloads cannot reconstruct original
bytes, so the binary protocol has not been identified.

The recorder now accepts only printable ASCII SDK acknowledgements (ok/error)
from 192.168.10.1:8889. Other command packets are retained as raw hex with source
port and acceptance status, and do not abort the eight-second wait. No binary
packet is interpreted as connection success. Seven mocked/offline tests passed.
Actual drone connectivity still needs a new lab recording. No flight commands
were added, and this package still records video without a live preview.
