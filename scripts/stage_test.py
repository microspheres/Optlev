import ctypes

nf = ctypes.WinDLL(r"C:\Program Files\New Focus\New Focus Picomotor Application\Samples\CmdLib8472.dll")

cmdLib = nf.CmdLib8742(True, 5000, strDeviceKey)
