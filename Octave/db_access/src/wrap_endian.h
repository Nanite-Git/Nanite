/*

This file is in the public domain. The Apple-specific include and
defines were taken from a public domain file at this site:

https://gist.github.com/yinyin/2027912

*/

#ifndef __OCT_PQ_WRAP_ENDIAN__

#define __OCT_PQ_WRAP_ENDIAN__

#ifdef __APPLE__

#include <libkern/OSByteOrder.h>
 
#define htobe16(x) OSSwapHostToBigInt16(x)
#define htole16(x) OSSwapHostToLittleInt16(x)
#define be16toh(x) OSSwapBigToHostInt16(x)
#define le16toh(x) OSSwapLittleToHostInt16(x)
 
#define htobe32(x) OSSwapHostToBigInt32(x)
#define htole32(x) OSSwapHostToLittleInt32(x)
#define be32toh(x) OSSwapBigToHostInt32(x)
#define le32toh(x) OSSwapLittleToHostInt32(x)
 
#define htobe64(x) OSSwapHostToBigInt64(x)
#define htole64(x) OSSwapHostToLittleInt64(x)
#define be64toh(x) OSSwapBigToHostInt64(x)
#define le64toh(x) OSSwapLittleToHostInt64(x)
 
#else

#ifdef __MINGW32__
#include <winsock2.h>

#define htobe16(x) htons(x)
#define htole16(x) (x)
#define be16toh(x) ntohs(x)
#define le16toh(x) (x)
 
#define htobe32(x) htonl(x)
#define htole32(x) (x)
#define be32toh(x) ntohl(x)
#define le32toh(x) (x)
 
#define htobe64(x) (((uint64_t)htonl((uint32_t)(x>>32)))|(((uint64_t)htonl(x&0xFFFFFFFF))<<32LL))
#define htole64(x) (x)
#define be64toh(x) (((uint64_t)ntohl((uint32_t)(x>>32)))|(((uint64_t)ntohl(x&0xFFFFFFFF))<<32LL))
#define le64toh(x) (x)
 
#else
#include <endian.h>
#endif

#endif

#endif // __OCT_PQ_WRAP_ENDIAN__
