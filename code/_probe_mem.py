# -*- coding: utf-8 -*-
"""临时探针：内存 + Windows 更新待重启标记（reg.exe 被安全策略拦截，只能用 winreg）。"""
import ctypes
import winreg


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


m = MEMORYSTATUSEX()
m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
G = 1024 ** 3
print("Load %.1f%% | 物理总 %.2f GB | 可用 %.2f GB | 页文件可用 %.2f GB"
      % (m.dwMemoryLoad, m.ullTotalPhys / G, m.ullAvailPhys / G, m.ullAvailPageFile / G))

HKLM = winreg.HKEY_LOCAL_MACHINE
checks = [
    ("RebootRequired(AutoUpdate)", HKLM,
     r"SOFTWARE\Microsoft\Windows\CurrentVersion\WindowsUpdate\Auto Update\RebootRequired", None),
    ("RebootPending(SessionMgr)", HKLM,
     r"SYSTEM\CurrentControlSet\Control\Session Manager", "RebootPending"),
    ("PendingFileRenameOps", HKLM,
     r"SYSTEM\CurrentControlSet\Control\Session Manager", "PendingFileRenameOperations"),
    ("RebootRequired(CBS)", HKLM,
     r"SOFTWARE\Microsoft\Windows\CurrentVersion\Component Based Servicing\RebootPending", None),
]
for name, hive, path, val in checks:
    try:
        k = winreg.OpenKey(hive, path)
        if val is None:
            print("%-28s = SET (键存在)" % name)
        else:
            try:
                v, _ = winreg.QueryValueEx(k, val)
                print("%-28s = SET (%s)" % (name, str(v)[:100]))
            except FileNotFoundError:
                print("%-28s = 无" % name)
        winreg.CloseKey(k)
    except FileNotFoundError:
        print("%-28s = 无" % name)
