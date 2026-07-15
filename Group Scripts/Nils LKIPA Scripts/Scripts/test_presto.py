# -*- coding: utf-8 -*-

from presto import lockin

# ADDRESS = "130.237.35.90" # from local desktop
# ADDRESS = "alfa"  # from Dell laptop
# PORT = 42871
# PORT = 42873 #from local desktop
ADDRESS = "192.168.88.51"
# ADDRESS = "192.168.88.54"
# ADDRESS = "192.168.88.50"

# with lockin.Lockin(address = ADDRESS, port = PORT) as lck: #from local desktop
with lockin.Lockin(address=ADDRESS) as lck:  # from Dell laptop
    print("Connected")
    print(f"fs: {lck.get_fs('adc')} Hz")
