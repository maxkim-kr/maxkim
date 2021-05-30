/* vim: set tabstop=4:softtabstop=4:shiftwidth=4:ffs=unix */
#include <string.h>
#include <unistd.h>

#include "./main_internal.h"

#define PATH_CPU_USAGE "/proc/stat"
#define PATH_MEM_USAGE "/proc/meminfo"
#define MMIO_THERMAL_SENSOR 0xfed170f4

#define FAMILY_6 0x6
#define MODEL_APL 92
#define MODEL_EHL 134
#define EHL_MMIO_THERMAL_SENSOR 0xfec8597c

union thermal_sensor {
	struct
	{
		int32_t sa : 8;	// System agent
		int32_t iunit : 8; // Camera domain
		int32_t gt : 8;	// Graphics domain
		int32_t ia : 8;	// All IA cores
	};
	int32_t raw;
};

struct cpu_usage
{
	uint64_t user;
	uint64_t nice;
	uint64_t system;
	uint64_t idle;
	uint64_t iowait;
	uint64_t irq;
	uint64_t softirq;
};

/* Response template */
static msgpack_object usage_thermal[] = {
	MSGPACK_STATIC_INT(0), // System Agent
	MSGPACK_STATIC_INT(0), // Camera Domain
	MSGPACK_STATIC_INT(0), // Graphics Domain
	MSGPACK_STATIC_INT(0), // All IA Cores
};

static msgpack_object_kv usage_val[] = {
	{
		.key = MSGPACK_STATIC_STR("cpu"),
		.val.type = MSGPACK_OBJECT_ARRAY,
	},
	{
		.key = MSGPACK_STATIC_STR("mem"),
		.val.type = MSGPACK_OBJECT_FLOAT,
	},
	{
		.key = MSGPACK_STATIC_STR("thermal"),
		.val = MSGPACK_STATIC_ARRAY(usage_thermal),
	},
};

static msgpack_object usage[] = {
	MSGPACK_STATIC_UINT(WS_RPC_CUSTOM_RESPONSE_ID),
	MSGPACK_STATIC_MAP(usage_val),
};

static msgpack_object result = MSGPACK_STATIC_ARRAY(usage);

/* Global monitor */
static struct
{
	// per CPU data
	int cpu_count;
	struct cpu_usage *cpu_usage;

	struct periodic_work *pw;
	int fd_cpu;
	int fd_mem;
	struct mmio mmio_thermal;
} MONITOR = {
	.cpu_count = -1,
	.fd_cpu = -1,
	.fd_mem = -1,
	.mmio_thermal = {.flags = MMIO_FLAG_PINNED},
};

/* CPU Usage Helper */
typedef int (*cpu_usage_cb_t)(int cpu, struct cpu_usage *cu);

static inline int parse_cpu_usage(char *buf, struct cpu_usage *cu)
{
	return (sscanf(buf, "cpu%*d %lu %lu %lu %lu %lu %lu %lu%*s", &cu->user,
				   &cu->nice, &cu->system, &cu->idle, &cu->iowait, &cu->irq,
				   &cu->softirq) < 7)
			   ? -1
			   : 0;
}

static inline double calculate_cpu_usage(const struct cpu_usage *prev,
										 const struct cpu_usage *curr)
{
	uint64_t cpu_used =
		(curr->user - prev->user) + (curr->nice - prev->nice) +
		(curr->system - prev->system) + (curr->iowait - prev->iowait) +
		(curr->irq - prev->irq) + (curr->softirq - prev->softirq);
	uint64_t cpu_total = cpu_used + (curr->idle - prev->idle);

	return cpu_used * 100.0 / cpu_total;
}

static inline int for_each_cpu_usage(cpu_usage_cb_t cb)
{
	char buf[4096];
	char *token, *p;
	int ret, idx;

	if ((ret = pread(MONITOR.fd_cpu, buf, (MONITOR.cpu_count + 1) << 6, 0)) < 0)
		ret = 0;
	buf[ret] = '\0';

	struct cpu_usage cu;
	for (ret = idx = 0, p = buf;
		 idx <= MONITOR.cpu_count && (token = strtok_r(p, "\n", &p)) != NULL;
		 idx++)
	{
		if (idx == 0)
			continue;

		if (parse_cpu_usage(token, &cu) != 0)
			continue;

		if ((ret = cb(idx - 1, &cu)) < 0)
			break;
	}

	return ret;
}

static int load_initial_cpu_usage(int cpu, struct cpu_usage *cu)
{
	return MONITOR.cpu_usage[cpu] = *cu, 0;
}

static int fill_cpu_usage(int cpu_idx, struct cpu_usage *cu)
{
	msgpack_object *cpu = &usage_val[0].val.via.array.ptr[cpu_idx];
	cpu->via.f64 = calculate_cpu_usage(&MONITOR.cpu_usage[cpu_idx], cu);
	MONITOR.cpu_usage[cpu_idx] = *cu;
	return 0;
}

static inline int fill_mem_usage(void)
{
	char buf[256];
	int ret;

	if ((ret = pread(MONITOR.fd_mem, buf, sizeof(buf), 0)) < 0)
		ret = 0;
	buf[ret] = '\0';

	int64_t mem_total = -1, mem_free = -1;
	for (char *token, *p = buf; (token = strtok_r(p, "\n", &p)) != NULL;)
	{
		if (strncmp(token, "MemTotal:", 9) == 0)
			sscanf(token, "MemTotal: %lu", &mem_total);
		else if (strncmp(token, "MemFree:", 8) == 0)
			sscanf(token, "MemFree: %lu", &mem_free);

		if (mem_total != -1 && mem_free != -1)
			break;
	}

	if (mem_total == -1 || mem_free == -1)
		return -1;

	usage_val[1].val.via.f64 = 100.0 - (mem_free * 100.0 / mem_total);
	return 0;
}

static int fill_thermal(void)
{
	union thermal_sensor thermal = { 0 };

	if (MONITOR.mmio_thermal.addr)
	{
		// thermal
		// TODO: Temperally divide the EHL/APL to get the temperature
		const struct processor_version_info *info =
			cpuid_processor_version_info();
		if (info->_family_id == FAMILY_6 && info->model != MODEL_APL)
		{
			// Elkhart Lake
			thermal.ia =
				mmio_get8(&MONITOR.mmio_thermal, EHL_MMIO_THERMAL_SENSOR);
			thermal.gt = mmio_get8(&MONITOR.mmio_thermal,
								EHL_MMIO_THERMAL_SENSOR + 0x4);
		}
		else
		{
			// Apollo Lake
			thermal.raw =
				mmio_get32(&MONITOR.mmio_thermal, MMIO_THERMAL_SENSOR);
		}
	}

	usage_thermal[0].via.i64 = thermal.sa;
	usage_thermal[1].via.i64 = thermal.iunit;
	usage_thermal[2].via.i64 = thermal.gt;
	usage_thermal[3].via.i64 = thermal.ia;
}

static int monitor_thread(void *work, const void *work_prev)
{
	struct ws_ctx *ctx = (struct ws_ctx *)work;
	const struct ws_ctx *ctx_prev = (const struct ws_ctx *)work_prev;

	// first time of processing
	log_info("Work(%p), Prev. Work(%p)", work, work_prev);
	if (work_prev == NULL)
	{
		// CPU usage
		for_each_cpu_usage(fill_cpu_usage);

		// Mem usage
		fill_mem_usage();

		// Thermal
		fill_thermal();
	}

	// build response
	struct ws_buffer buf;
	ws_buffer_init(&buf, 0);
	msgpack_packer *pk = msgpack_packer_new(&buf, ws_msgpack_simple_writer);
	msgpack_pack_object(pk, result);
	if (ws_commit_buffer(ctx, &buf, WS_WRITE_BINARY) < 0)
		ws_buffer_destroy(&buf);
	msgpack_packer_free(pk);
	return 0;
}

int main_monitor_init(void)
{
	MONITOR.cpu_count = cpuid_core_count();
	msgpack_object *cpu =
		(msgpack_object *)malloc(sizeof(msgpack_object) * MONITOR.cpu_count);
	if (cpu == NULL)
		goto failed;

	MONITOR.cpu_usage = (struct cpu_usage *)malloc(sizeof(struct cpu_usage) *
												   MONITOR.cpu_count);
	if (MONITOR.cpu_usage == NULL)
		goto failed;

	for (int i = 0; i < MONITOR.cpu_count; i++)
	{
		cpu[i].type = MSGPACK_OBJECT_FLOAT;
		cpu[i].via.f64 = 0;
	}

	usage_val[0].val.via.array.size = MONITOR.cpu_count;
	usage_val[0].val.via.array.ptr = cpu;

	MONITOR.fd_cpu = open(PATH_CPU_USAGE, O_RDONLY);
	if (MONITOR.fd_cpu < 0)
		goto failed;

	// Initial CPU data
	for_each_cpu_usage(load_initial_cpu_usage);

	MONITOR.fd_mem = open(PATH_MEM_USAGE, O_RDONLY);
	if (MONITOR.fd_mem < 0)
		goto failed;

	// TODO: get phy address from configuration
	const struct processor_version_info *info = cpuid_processor_version_info();
	off64_t addr = (info->family_id == FAMILY_6 && info->model != MODEL_APL)
					   ? EHL_MMIO_THERMAL_SENSOR
					   : MMIO_THERMAL_SENSOR;
	if (mmio_map_page(&MONITOR.mmio_thermal, addr) != 0) {
		// Don't report thermal if not supported
		MONITOR.mmio_thermal.addr = NULL;
	}

	MONITOR.pw = periodic_work_create(-1, monitor_thread, NULL);
	if (MONITOR.pw == NULL)
		goto failed;

	return 0;

failed:
	main_monitor_destroy();
	return -1;
}

int main_monitor_destroy(void)
{
	if (MONITOR.pw != NULL)
		periodic_work_destroy(MONITOR.pw);

	if (usage_val[0].val.via.array.ptr != NULL)
		free(usage_val[0].val.via.array.ptr);

	if (MONITOR.cpu_usage != NULL)
		free(MONITOR.cpu_usage);

	if (MONITOR.fd_cpu != -1)
		close(MONITOR.fd_cpu);

	if (MONITOR.fd_mem != -1)
		close(MONITOR.fd_mem);

	if (MONITOR.mmio_thermal.addr)
		mmio_unmap_page(&MONITOR.mmio_thermal);

	return 0;
}

int main_monitor_connect(struct ws_ctx *ctx, useconds_t interval_us)
{
	return periodic_work_add(MONITOR.pw, interval_us, ctx);
}

int main_monitor_disconnect(struct ws_ctx *ctx)
{
	return periodic_work_del(MONITOR.pw, ctx);
}
