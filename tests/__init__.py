import logging

# Failover tests trigger expected warnings; keep test output readable.
logging.getLogger("flotilla").setLevel(logging.ERROR)
