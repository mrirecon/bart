
#ifndef BART_MISC_DLLSPEC
#define BART_MISC_DLLSPEC

#if defined(_WIN64) || defined(_WIN32)

#ifdef BARTLIB_EXPORTS

// Building a DLL
#define BARTLIB_API __declspec(dllexport)
#else

#ifdef BARTLIB_STATIC

// Building/Using a MS static lib
#define BARTLIB_API
#else

// Using a dll
#define BARTLIB_API __declspec(dllimport)
#endif
#endif

#define BARTLIB_CALL __cdecl
#else

#define BARTLIB_API
#define BARTLIB_CALL
#endif

#endif

