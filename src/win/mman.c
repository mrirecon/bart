/*
MIT licence

Copyright (c) 2021 Tamás Hakkel <hakkelt@gmail.com>
Copyright (c) 2013-2019 Steven Lee
Copyright (c) 2010-2012 Viktor Kutuzov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
 */

#include <windows.h>
#include <fcntl.h>
#include <errno.h>

#include "misc/debug.h"
#define BUF_SIZE 256

#include "misc/misc.h"
#include "win/mman.h"

// Reference: https://docs.microsoft.com/en-us/windows/win32/memory/creating-named-shared-memory
int shm_open(const char *name, int oflag, mode_t mode)
{
	UNUSED(mode);
	HANDLE hMapFile;
	if ((oflag & _O_CREAT) == 0) {
		hMapFile = CreateFileMapping(
							INVALID_HANDLE_VALUE,      // use paging file
							NULL,                      // default security
							PAGE_READWRITE,            // read/write access
							0,                         // maximum object size (high-order DWORD)
							BUF_SIZE,                  // maximum object size (low-order DWORD)
							strcat("Global\\", name)); // name of mapping object
	} else {
		hMapFile = OpenFileMapping(
					FILE_MAP_ALL_ACCESS,               // read/write access
					FALSE,                             // do not inherit the name
					strcat("Global\\", name));         // name of mapping object
	}

	if (hMapFile == NULL)
		return GetLastError();

	return _open_osfhandle((intptr_t)hMapFile, PAGE_READWRITE);
}

int shm_unlink(const char *name)
{
	UNUSED(name);
	return 0; // Handled automatically by the OS
}

/* The code below was adapted from GitHub: https://github.com/alitrack/mman-win32 */

#include <io.h>
#include <stdbool.h>

#ifndef FILE_MAP_EXECUTE
#define FILE_MAP_EXECUTE    0x0020
#endif /* FILE_MAP_EXECUTE */

// Map Windows error codes (https://docs.microsoft.com/en-us/windows/win32/debug/system-error-codes)
// to POSIX error codes (https://man7.org/linux/man-pages/man3/errno.3.html)
static int __map_mman_error(const DWORD err, const int deferr)
{
    if (err == 0)
        return 0;
    //TODO: implement
    return err;
}

static DWORD __map_mmap_prot_page(const int prot, const bool write_copy)
{
	DWORD protect = 0;
	
	if (prot == PROT_NONE)
		return protect;
		
	if ((prot & PROT_EXEC) != 0) {
		protect = ((prot & PROT_WRITE) != 0) ? 
					PAGE_EXECUTE_READWRITE : (write_copy ? PAGE_EXECUTE_WRITECOPY : PAGE_EXECUTE_READ);
	}
	else {
		protect = ((prot & PROT_WRITE) != 0) ?
					(write_copy ? PAGE_WRITECOPY : PAGE_READWRITE) : PAGE_READONLY;
	}
	
	return protect;
}

static DWORD __map_mmap_prot_file(const int prot, const bool write_copy)
{
	DWORD desiredAccess = 0;
	
	if (prot == PROT_NONE)
		return desiredAccess;
	
	if (write_copy) {
		if ((prot & PROT_WRITE) != 0)
			// Note: Documentation is incorrect (https://docs.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-mapviewoffileex)
			// using FILE_MAP_COPY|FILE_MAP_READ (as it is recommended in the documentation) leads to segmentation fault on writes
			// https://stackoverflow.com/questions/55018806/copy-on-write-file-mapping-on-windows
			desiredAccess |= FILE_MAP_COPY;
		else if ((prot & PROT_READ) != 0)
			desiredAccess |= FILE_MAP_READ;
	} else {
		if ((prot & PROT_READ) != 0)
			desiredAccess |= FILE_MAP_READ;
		if ((prot & PROT_WRITE) != 0)
			desiredAccess |= FILE_MAP_WRITE;
	}
	
	if ((prot & PROT_EXEC) != 0)
		desiredAccess |= FILE_MAP_EXECUTE;
	
	return desiredAccess;
}

void* mmap(void *addr, size_t len, int prot, int flags, int fildes, OffsetType off)
{
	HANDLE fm, h;
	
	void * map = MAP_FAILED;
	
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4293)
#endif

	const DWORD dwFileOffsetLow = (sizeof(OffsetType) <= sizeof(DWORD)) ?
					(DWORD)off : (DWORD)(off & 0xFFFFFFFFL);
	const DWORD dwFileOffsetHigh = (sizeof(OffsetType) <= sizeof(DWORD)) ?
					(DWORD)0 : (DWORD)((off >> 32) & 0xFFFFFFFFL);

	const OffsetType maxSize = off + (OffsetType)len;

	const DWORD dwMaxSizeLow = (sizeof(OffsetType) <= sizeof(DWORD)) ?
					(DWORD)maxSize : (DWORD)(maxSize & 0xFFFFFFFFL);
	const DWORD dwMaxSizeHigh = (sizeof(OffsetType) <= sizeof(DWORD)) ?
					(DWORD)0 : (DWORD)((maxSize >> 32) & 0xFFFFFFFFL);
	
	DWORD protect = __map_mmap_prot_page(prot, false);
	DWORD desiredAccess = __map_mmap_prot_file(prot, false);

#ifdef _MSC_VER
#pragma warning(pop)
#endif

	errno = 0;
	
	if (len == 0 /* Usupported protection combinations */ || prot == PROT_EXEC) {
		errno = EINVAL;
		return MAP_FAILED;
	}
	
	h = ((flags & MAP_ANONYMOUS) == 0) ? 
					(HANDLE)_get_osfhandle(fildes) : INVALID_HANDLE_VALUE;

	if ((flags & MAP_ANONYMOUS) == 0 && h == INVALID_HANDLE_VALUE) {
		errno = EBADF;
		return MAP_FAILED;
	}

	fm = CreateFileMapping(h, NULL, protect, dwMaxSizeHigh, dwMaxSizeLow, NULL);

	if (fm == NULL) {
		DWORD error_code = GetLastError();
		if (error_code == ERROR_ACCESS_DENIED && (prot & PROT_WRITE) != 0) {
			// ERROR_ACCESS_DENIED can occur if file opened read-only,
			// and we want a read-write mapping that doesn't modify the file on the disk.
			// Therefore let's try again with WRITE_COPY flags:
			protect = __map_mmap_prot_page(prot, true);
			desiredAccess = __map_mmap_prot_file(prot, true);

			fm = CreateFileMapping(h, NULL, protect, dwMaxSizeHigh, dwMaxSizeLow, NULL);

			if (fm == NULL) {
				errno = __map_mman_error(GetLastError(), EPERM);
				return MAP_FAILED;
			}
		} else {
			errno = __map_mman_error(error_code, EPERM);
			return MAP_FAILED;
		}
	}
  
	if ((flags & MAP_FIXED) == 0) {
		map = MapViewOfFile(fm, desiredAccess, dwFileOffsetHigh, dwFileOffsetLow, len);
	} else {
		map = MapViewOfFileEx(fm, desiredAccess, dwFileOffsetHigh, dwFileOffsetLow, len, addr);
	}

	CloseHandle(fm);
  
	if (map == NULL) {
		errno = __map_mman_error(GetLastError(), EPERM);
		return MAP_FAILED;
	}

	return map;
}

int munmap(void *addr, size_t len)
{
	UNUSED(len);

	if (UnmapViewOfFile(addr))
		return 0;
		
	int error = GetLastError();
	
	// In POSIX, munmap is supposed to throw no errors when trying to unmap a memory address that is not a mapped memory segment.
	// On the other hand, Windows throws an error in this case that can be ignored when emulating functionality of munmap.
	if (error == ERROR_INVALID_ADDRESS)
		return 0;
	
	errno =  __map_mman_error(error, EPERM);
	
	return -1;
}

int _mprotect(void *addr, size_t len, int prot)
{
	DWORD newProtect = __map_mmap_prot_page(prot, false);
	DWORD oldProtect = 0;
	
	if (VirtualProtect(addr, len, newProtect, &oldProtect))
		return 0;
	
	errno =  __map_mman_error(GetLastError(), EPERM);
	
	return -1;
}

int msync(void *addr, size_t len, int flags)
{
	if ((flags & MS_SYNC) != 0 || (flags & MS_INVALIDATE) != 0)
		error("MS_SYNC and MS_INVALIDATE are not supported on Windows.\n");

	if (FlushViewOfFile(addr, len))
		return 0;
	
	errno =  __map_mman_error(GetLastError(), EPERM);
	
	return -1;
}

int mlock(const void *addr, size_t len)
{
	if (VirtualLock((LPVOID)addr, len))
		return 0;
		
	errno =  __map_mman_error(GetLastError(), EPERM);
	
	return -1;
}

int munlock(const void *addr, size_t len)
{
	if (VirtualUnlock((LPVOID)addr, len))
		return 0;
		
	errno =  __map_mman_error(GetLastError(), EPERM);
	
	return -1;
}
