/* vim: set tabstop=4:softtabstop=4:shiftwidth=4:ffs=unix */
#ifndef _MAIN_INTERNAL_H_
#define _MAIN_INTERNAL_H_

#include <fcntl.h>
#include <unistd.h>

#include <dtdt/core.h>
#include <dtdt/drivers/cpuid.h>
#include <dtdt/drivers/mmio.h>

enum MAIN_PROTOCOL_ERROR
{
	MAINP_ERROR = DTDT_ERROR_PROTOCOL_ERROR_BASE,

	MAINP_ERROR_INVALID_INTERVAL,
	MAINP_ERROR_INVALID_PASSWORD,
	MAINP_ERROR_PASSWORD_MISMATCH,
	MAINP_ERROR_UNABLE_TO_STOP_USAGE,

	MAINP_ERROR_NOT_ENOUGH_MEMORY,
	MAINP_ERROR_INTERNAL_ERROR,
};

#define MSGPACK_STATIC_STR(name)                                               \
	{                                                                          \
		.type = MSGPACK_OBJECT_STR, .via.str.ptr = name,                       \
		.via.str.size = sizeof(name) - 1,                                      \
	}

#define MSGPACK_STATIC_ARRAY(_arr)                                             \
	{                                                                          \
		.type = MSGPACK_OBJECT_ARRAY, .via.array.ptr = _arr,                   \
		.via.array.size = countof(_arr),                                       \
	}

#define MSGPACK_STATIC_MAP(_map)                                               \
	{                                                                          \
		.type = MSGPACK_OBJECT_MAP, .via.map.ptr = _map,                       \
		.via.map.size = countof(_map),                                         \
	}

#define MSGPACK_STATIC_INT(val)                                                \
	{                                                                          \
		.type = MSGPACK_OBJECT_NEGATIVE_INTEGER, .via.i64 = val,               \
	}

#define MSGPACK_STATIC_UINT(val)                                               \
	{                                                                          \
		.type = MSGPACK_OBJECT_POSITIVE_INTEGER, .via.u64 = val,               \
	}

#define MSGPACK_STATIC_DOUBLE(val)                                             \
	{                                                                          \
		.type = MSGPACK_OBJECT_FLOAT, .via.f64 = val,                          \
	}

int main_monitor_init(void);
int main_monitor_destroy(void);
int main_monitor_connect(struct ws_ctx *ctx, useconds_t interval_us);
int main_monitor_disconnect(struct ws_ctx *ctx);

/* Helper APIs */
static inline size_t read_file(const char *path, char *buf, long buf_size)
{
	int fd = open(path, O_RDONLY);
	if (fd == -1)
		return 0;

	size_t size = read(fd, buf, buf_size);
	close(fd);
	return size;
}

static inline uint32_t buf_to_uint32(char *buf)
{
	return (buf[3] << 24 | buf[2] << 16 | buf[1] << 8 | buf[0]);
}

static inline uint64_t addr_from_iomem(const char *str)
{
	// assume that the buffer size doesn't exceed 8192
	char buf[8192];
	char *line, *s;
	uint64_t addr = 0;
	int fd = open("/proc/iomem", O_RDONLY);
	if (fd == -1)
		return 0;

	ssize_t len = 0, count = 0;
	do
	{
		len = read(fd, buf + count, 4096);
		count += len;
	} while (len > 0);

	line = strtok(buf, "\n");
	while (line != NULL)
	{
		s = strstr(line, str);
		if (s != NULL)
		{
			sscanf(line, "%lx-", &addr);
			return addr;
		}
		line = strtok(NULL, "\n");
	}

	close(fd);

	return addr;
}

#endif /* _MAIN_INTERNAL_H_ */
