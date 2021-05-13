/* Authors:
 * 2021 Tamás Hakkel <hakkelt@gmail.com>
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

#ifndef EBADRQC
#define EBADRQC 54	    /* Invalid request code */
#endif /* EBADRQC */
#ifndef ENONET
#define ENONET 64	    /* Machine is not on the network */
#endif /* ENONET */
#ifndef ECOMM
#define	ECOMM 70	    /* Communication error on send */
#endif /* ECOMM */
#ifndef ENOTUNIQ
#define ENOTUNIQ 80	    /* Given log. name not unique */
#endif /* ENOTUNIQ */
#ifndef ELIBBAD
#define ELIBBAD 84	    /* Accessing a corrupted shared lib */
#endif /* ELIBBAD */
#ifndef ENMFILE
#define ENMFILE 89      /* No more files */
#endif /* ENMFILE */
#ifndef ENOMEDIUM
#define ENOMEDIUM 135   /* No medium (in tape drive) */
#endif /* ENOMEDIUM */

// Map Windows error codes (https://docs.microsoft.com/en-us/windows/win32/debug/system-error-codes)
// to POSIX error codes (https://man7.org/linux/man-pages/man3/errno.3.html)
static int __map_mman_error(const DWORD err, const int deferr)
{
	if (err == 0)
		return 0;

	// That mapping is adapted from the Newlib-Cygwin project:
	// https://github.com/mirror/newlib-cygwin/blob/70782484855f3ecedfc9c3caf5397b380c0328b8/winsup/cygwin/errno.cc
	switch (err) {
 		case ERROR_ACCESS_DENIED:
			return EACCES;
 		case ERROR_ACTIVE_CONNECTIONS:
			return EAGAIN;
 		case ERROR_ALREADY_EXISTS:
			return EEXIST;
 		case ERROR_BAD_DEVICE:
			return ENODEV;
 		case ERROR_BAD_EXE_FORMAT:
			return ENOEXEC;
 		case ERROR_BAD_NETPATH:
			return ENOENT;
 		case ERROR_BAD_NET_NAME:
			return ENOENT;
 		case ERROR_BAD_NET_RESP:
			return ENOSYS;
 		case ERROR_BAD_PATHNAME:
			return ENOENT;
 		case ERROR_BAD_PIPE:
			return EINVAL;
 		case ERROR_BAD_UNIT:
			return ENODEV;
 		case ERROR_BAD_USERNAME:
			return EINVAL;
 		case ERROR_BEGINNING_OF_MEDIA:
			return EIO;
 		case ERROR_BROKEN_PIPE:
			return EPIPE;
 		case ERROR_BUSY:
			return EBUSY;
 		case ERROR_BUS_RESET:
			return EIO;
 		case ERROR_CALL_NOT_IMPLEMENTED:
			return ENOSYS;
 		case ERROR_CANCELLED:
			return EINTR;
 		case ERROR_CANNOT_MAKE:
			return EPERM;
 		case ERROR_CHILD_NOT_COMPLETE:
			return EBUSY;
 		case ERROR_COMMITMENT_LIMIT:
			return EAGAIN;
 		case ERROR_CONNECTION_REFUSED:
			return ECONNREFUSED;
 		case ERROR_CRC:
			return EIO;
 		case ERROR_DEVICE_DOOR_OPEN:
			return EIO;
 		case ERROR_DEVICE_IN_USE:
			return EAGAIN;
 		case ERROR_DEVICE_REQUIRES_CLEANING:
			return EIO;
 		case ERROR_DEV_NOT_EXIST:
			return ENOENT;
 		case ERROR_DIRECTORY:
			return ENOTDIR;
 		case ERROR_DIR_NOT_EMPTY:
			return ENOTEMPTY;
 		case ERROR_DISK_CORRUPT:
			return EIO;
 		case ERROR_DISK_FULL:
			return ENOSPC;
 		case ERROR_DS_GENERIC_ERROR:
			return EIO;
 		case ERROR_DUP_NAME:
			return ENOTUNIQ;
 		case ERROR_EAS_DIDNT_FIT:
			return ENOSPC;
 		case ERROR_EAS_NOT_SUPPORTED:
			return ENOTSUP;
 		case ERROR_EA_LIST_INCONSISTENT:
			return EINVAL;
 		case ERROR_EA_TABLE_FULL:
			return ENOSPC;
 		case ERROR_END_OF_MEDIA:
			return ENOSPC;
 		case ERROR_EOM_OVERFLOW:
			return EIO;
 		case ERROR_EXE_MACHINE_TYPE_MISMATCH:
			return ENOEXEC;
 		case ERROR_EXE_MARKED_INVALID:
			return ENOEXEC;
 		case ERROR_FILEMARK_DETECTED:
			return EIO;
 		case ERROR_FILENAME_EXCED_RANGE:
			return ENAMETOOLONG;
 		case ERROR_FILE_CORRUPT:
			return EEXIST;
 		case ERROR_FILE_EXISTS:
			return EEXIST;
 		case ERROR_FILE_INVALID:
			return ENXIO;
 		case ERROR_FILE_NOT_FOUND:
			return ENOENT;
 		case ERROR_HANDLE_DISK_FULL:
			return ENOSPC;
 		case ERROR_HANDLE_EOF:
			return ENODATA;
 		case ERROR_INVALID_ADDRESS:
			return EINVAL;
 		case ERROR_INVALID_AT_INTERRUPT_TIME:
			return EINTR;
 		case ERROR_INVALID_BLOCK_LENGTH:
			return EIO;
 		case ERROR_INVALID_DATA:
			return EINVAL;
 		case ERROR_INVALID_DRIVE:
			return ENODEV;
 		case ERROR_INVALID_EA_NAME:
			return EINVAL;
 		case ERROR_INVALID_EXE_SIGNATURE:
			return ENOEXEC;
 		case ERROR_INVALID_FUNCTION:
			return EBADRQC;
 		case ERROR_INVALID_HANDLE:
			return EBADF;
 		case ERROR_INVALID_NAME:
			return ENOENT;
 		case ERROR_INVALID_PARAMETER:
			return EINVAL;
 		case ERROR_INVALID_SIGNAL_NUMBER:
			return EINVAL;
 		case ERROR_IOPL_NOT_ENABLED:
			return ENOEXEC;
 		case ERROR_IO_DEVICE:
			return EIO;
 		case ERROR_IO_INCOMPLETE:
			return EAGAIN;
 		case ERROR_IO_PENDING:
			return EAGAIN;
 		case ERROR_LOCK_VIOLATION:
			return EBUSY;
 		case ERROR_MAX_THRDS_REACHED:
			return EAGAIN;
 		case ERROR_META_EXPANSION_TOO_LONG:
			return EINVAL;
 		case ERROR_MOD_NOT_FOUND:
			return ENOENT;
 		case ERROR_MORE_DATA:
			return EMSGSIZE;
 		case ERROR_NEGATIVE_SEEK:
			return EINVAL;
 		case ERROR_NETNAME_DELETED:
			return ENOENT;
 		case ERROR_NOACCESS:
			return EFAULT;
 		case ERROR_NONE_MAPPED:
			return EINVAL;
 		case ERROR_NONPAGED_SYSTEM_RESOURCES:
			return EAGAIN;
 		case ERROR_NOT_CONNECTED:
			return ENOLINK;
 		case ERROR_NOT_ENOUGH_MEMORY:
			return ENOMEM;
 		case ERROR_NOT_ENOUGH_QUOTA:
			return EIO;
 		case ERROR_NOT_OWNER:
			return EPERM;
 		case ERROR_NOT_READY:
			return ENOMEDIUM;
 		case ERROR_NOT_SAME_DEVICE:
			return EXDEV;
 		case ERROR_NOT_SUPPORTED:
			return ENOSYS;
 		case ERROR_NO_DATA:
			return EPIPE;
 		case ERROR_NO_DATA_DETECTED:
			return EIO;
 		case ERROR_NO_MEDIA_IN_DRIVE:
			return ENOMEDIUM;
 		case ERROR_NO_MORE_FILES:
			return ENMFILE;
 		case ERROR_NO_MORE_ITEMS:
			return ENMFILE;
 		case ERROR_NO_MORE_SEARCH_HANDLES:
			return ENFILE;
 		case ERROR_NO_PROC_SLOTS:
			return EAGAIN;
 		case ERROR_NO_SIGNAL_SENT:
			return EIO;
 		case ERROR_NO_SYSTEM_RESOURCES:
			return EFBIG;
 		case ERROR_NO_TOKEN:
			return EINVAL;
 		case ERROR_OPEN_FAILED:
			return EIO;
 		case ERROR_OPEN_FILES:
			return EAGAIN;
 		case ERROR_OUTOFMEMORY:
			return ENOMEM;
 		case ERROR_PAGED_SYSTEM_RESOURCES:
			return EAGAIN;
 		case ERROR_PAGEFILE_QUOTA:
			return EAGAIN;
 		case ERROR_PATH_NOT_FOUND:
			return ENOENT;
 		case ERROR_PIPE_BUSY:
			return EBUSY;
 		case ERROR_PIPE_CONNECTED:
			return EBUSY;
 		case ERROR_PIPE_LISTENING:
			return ECOMM;
 		case ERROR_PIPE_NOT_CONNECTED:
			return ECOMM;
 		case ERROR_POSSIBLE_DEADLOCK:
			return EDEADLOCK;
 		case ERROR_PRIVILEGE_NOT_HELD:
			return EPERM;
 		case ERROR_PROCESS_ABORTED:
			return EFAULT;
 		case ERROR_PROC_NOT_FOUND:
			return ESRCH;
 		case ERROR_REM_NOT_LIST:
			return ENONET;
 		case ERROR_SECTOR_NOT_FOUND:
			return EINVAL;
 		case ERROR_SEEK:
			return EINVAL;
 		case ERROR_SERVICE_REQUEST_TIMEOUT:
			return EBUSY;
 		case ERROR_SETMARK_DETECTED:
			return EIO;
 		case ERROR_SHARING_BUFFER_EXCEEDED:
			return ENOLCK;
 		case ERROR_SHARING_VIOLATION:
			return EBUSY;
 		case ERROR_SIGNAL_PENDING:
			return EBUSY;
 		case ERROR_SIGNAL_REFUSED:
			return EIO;
 		case ERROR_SXS_CANT_GEN_ACTCTX:
			return ELIBBAD;
 		case ERROR_THREAD_1_INACTIVE:
			return EINVAL;
 		case ERROR_TIMEOUT:
			return EBUSY;
 		case ERROR_TOO_MANY_LINKS:
			return EMLINK;
 		case ERROR_TOO_MANY_OPEN_FILES:
			return EMFILE;
 		case ERROR_UNEXP_NET_ERR:
			return EIO;
 		case ERROR_WAIT_NO_CHILDREN:
			return ECHILD;
 		case ERROR_WORKING_SET_QUOTA:
			return EAGAIN;
 		case ERROR_WRITE_PROTECT:
			return EROFS;
		default:
			return deferr;
	}
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
