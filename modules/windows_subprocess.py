"""
Bind child process lifetimes to the parent.

On Windows, closing the console window or killing python.exe via Task Manager
doesn't deliver SIGTERM — signal handlers and atexit don't run, so subprocess
children are orphaned. A Job Object with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
makes the kernel reap any assigned children when the parent's handle is closed.
"""

import ctypes
import os

from modules.logging_colors import logger


_job_handle = None


class _BasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ('PerProcessUserTimeLimit', ctypes.c_int64),
        ('PerJobUserTimeLimit', ctypes.c_int64),
        ('LimitFlags', ctypes.c_uint32),
        ('MinimumWorkingSetSize', ctypes.c_size_t),
        ('MaximumWorkingSetSize', ctypes.c_size_t),
        ('ActiveProcessLimit', ctypes.c_uint32),
        ('Affinity', ctypes.c_size_t),
        ('PriorityClass', ctypes.c_uint32),
        ('SchedulingClass', ctypes.c_uint32),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [
        ('ReadOperationCount', ctypes.c_uint64),
        ('WriteOperationCount', ctypes.c_uint64),
        ('OtherOperationCount', ctypes.c_uint64),
        ('ReadTransferCount', ctypes.c_uint64),
        ('WriteTransferCount', ctypes.c_uint64),
        ('OtherTransferCount', ctypes.c_uint64),
    ]


class _ExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ('BasicLimitInformation', _BasicLimitInformation),
        ('IoInfo', _IoCounters),
        ('ProcessMemoryLimit', ctypes.c_size_t),
        ('JobMemoryLimit', ctypes.c_size_t),
        ('PeakProcessMemoryUsed', ctypes.c_size_t),
        ('PeakJobMemoryUsed', ctypes.c_size_t),
    ]


def _ensure_job():
    global _job_handle
    if _job_handle is not None:
        return _job_handle

    try:
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel32.CreateJobObjectW.restype = ctypes.c_void_p
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return None

        info = _ExtendedLimitInformation()
        info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            ctypes.c_void_p(job), 9,  # JobObjectExtendedLimitInformation
            ctypes.byref(info), ctypes.sizeof(info)
        ):
            kernel32.CloseHandle(ctypes.c_void_p(job))
            return None

        _job_handle = job
        return job
    except Exception:
        return None


def bind_to_parent_lifetime(pid):
    """Bind the given child process to this process's lifetime.

    When this process exits for any reason, the OS will clean up the child.
    No-op on non-Windows or if the Job Object cannot be set up.
    """
    if os.name != 'nt':
        return

    job = _ensure_job()
    if not job:
        return

    try:
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel32.OpenProcess.restype = ctypes.c_void_p
        handle = kernel32.OpenProcess(0x0001 | 0x0100, False, pid)  # TERMINATE | SET_QUOTA
        if not handle:
            return
        try:
            kernel32.AssignProcessToJobObject(ctypes.c_void_p(job), ctypes.c_void_p(handle))
        finally:
            kernel32.CloseHandle(ctypes.c_void_p(handle))
    except Exception as e:
        logger.debug(f"Could not bind child PID {pid} to parent lifetime: {e}")
