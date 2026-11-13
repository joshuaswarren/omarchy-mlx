/* SPDX-License-Identifier: (GPL-2.0-only WITH Linux-syscall-note) OR MIT */
/* Copyright 2022 Eileen Yoon <eyn@gmx.com> */
/*
 * Vendored for mlx-omarchy lane 3 (used under MIT) from aurora-silicon/linux
 * 172b882a62c54aa25db68b7f95b5c1be2a2150d0 include/uapi/drm/ane_accel.h,
 * sha256 77d2282438088886c7ca6cd80b5f1f15f63caddb1e8e86f28489f95275a28e73.
 * One edit: the "drm.h" include names <drm/drm.h> (linux-api-headers) so the
 * header builds outside a kernel tree. Replace it whole when the ABI moves;
 * never edit the layouts.
 */

#ifndef __DRM_UAPI_ANE_ACCEL_H__
#define __DRM_UAPI_ANE_ACCEL_H__

#include <drm/drm.h>

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * DOC: ABI versions
 *
 * Two drivers implement this interface on an accel node, both with the DRM
 * driver name "ane":
 *
 * - ane (H13 family: T8103, T6000, T6001, T6002) implements ABI version
 *   %DRM_ANE_ABI_V1: GET_CAPS, BO_INIT, BO_FREE and SUBMIT.
 * - ane_t6021 (H14 family: T6020, T6021, T6022, T8112) implements ABI
 *   version %DRM_ANE_ABI_V2: GET_CAPS, BO_INIT, BO_FREE, PROG_LOAD,
 *   PROC_CREATE and EXEC.
 * - ane_t6021 on H16 (T6041) and H17 (T8140) implements ABI version
 *   %DRM_ANE_ABI_V3: GET_CAPS, H16_LOAD and H16_EXEC. The node exists only
 *   after the staged opt-in reached its program stage. DRM_IOCTL_VERSION
 *   minor 1 adds surface mappings (%DRM_ANE_H16_MMAP_OFFSET) and
 *   %DRM_ANE_H16_EXEC_IN_PLACE.
 *
 * The DRM_IOCTL_VERSION major and &drm_ane_get_caps.abi_version both give
 * the ABI version. The two versions share the ioctl numbers and the
 * structures, and a number is never given to a second ioctl. A driver
 * rejects the ioctls of the other version. A structure grows only at its
 * end; &drm_ane_get_caps.size tells which fields the driver filled in.
 *
 * Every pad and reserved field, and every flags field with no flags
 * defined, must be zero. The drivers reject a nonzero value with -EINVAL.
 */
#define DRM_ANE_ABI_V1		1
#define DRM_ANE_ABI_V2		2
#define DRM_ANE_ABI_V3		3

#define DRM_ANE_CHIP_H13	13
#define DRM_ANE_CHIP_H14	14
#define DRM_ANE_CHIP_H16	16
#define DRM_ANE_CHIP_H17	17

/* ABI version 1: entries of &drm_ane_submit.handles */
#define DRM_ANE_TILE_COUNT	0x20
/* ABI version 1: network id of the queue that SUBMIT uses */
#define DRM_ANE_FIFO_NID	0x40
/* ABI version 1: the weights start at round_up(tsk_size, DRM_ANE_CMD_GRAN) */
#define DRM_ANE_CMD_GRAN	0x10

/* ABI version 2: highest section id, and most sections per PROG_LOAD */
#define DRM_ANE_MAX_SECTIONS	9
/* ABI version 2: most generic binds per PROG_LOAD and buffers per EXEC */
#define DRM_ANE_MAX_BINDS	64

#define DRM_ANE_GET_CAPS	0x00
#define DRM_ANE_BO_INIT		0x01
#define DRM_ANE_BO_FREE		0x02
#define DRM_ANE_SUBMIT		0x03
#define DRM_ANE_PROG_LOAD	0x04
#define DRM_ANE_PROC_CREATE	0x05
#define DRM_ANE_EXEC		0x06
#define DRM_ANE_H16_LOAD	0x07
#define DRM_ANE_H16_EXEC	0x08

/* ABI version 3: host-built program sections that H16_LOAD places */
#define DRM_ANE_H16_SECTION_GENERIC		1
#define DRM_ANE_H16_SECTION_OPERATION		2
#define DRM_ANE_H16_SECTION_PROCEDURE		3
#define DRM_ANE_H16_SECTION_TEXT_PROPERTIES	4
#define DRM_ANE_H16_SECTIONS			4
/* ABI version 3: smallest and largest host-built section, in bytes */
#define DRM_ANE_H16_SECTION_MIN			0x28
#define DRM_ANE_H16_SECTION_MAX			0x10000

/*
 * ABI version 3: firmware resource indices that the kernel gives the LOAD
 * slots. A host-built section that names a slot uses these values. In
 * particular, a program with a kernel section tags its operation record with
 * %DRM_ANE_H16_RESOURCE_KERNEL.
 */
#define DRM_ANE_H16_RESOURCE_GENERIC		1002
#define DRM_ANE_H16_RESOURCE_OPERATION		1003
#define DRM_ANE_H16_RESOURCE_PROCEDURE		1004
#define DRM_ANE_H16_RESOURCE_TEXT_PROPERTIES	1005
#define DRM_ANE_H16_RESOURCE_TEXT		1006
#define DRM_ANE_H16_RESOURCE_KERNEL		1007
/* ABI version 3: most surfaces of one program */
#define DRM_ANE_H16_SURFACES			4

/**
 * struct drm_ane_get_caps - DRM_IOCTL_ANE_GET_CAPS argument
 * @flags: In. No flags are defined: must be zero.
 * @size: Out. Bytes of this structure that the driver filled in.
 * @abi_version: Out. %DRM_ANE_ABI_V1, %DRM_ANE_ABI_V2 or %DRM_ANE_ABI_V3.
 * @chip_family: Out. %DRM_ANE_CHIP_H13, %DRM_ANE_CHIP_H14,
 *	%DRM_ANE_CHIP_H16 or %DRM_ANE_CHIP_H17.
 * @section_size: Out. Size of the &drm_ane_section entries that PROG_LOAD
 *	reads, or 0 if the driver has no PROG_LOAD.
 * @bind_size: Out. Size of the &drm_ane_generic_bind entries that PROG_LOAD
 *	reads, or 0 if the driver has no PROG_LOAD.
 * @exec_io_size: Out. Size of the &drm_ane_exec_io entries that EXEC reads,
 *	or 0 if the driver has no EXEC.
 * @pad: Must be zero.
 */
struct drm_ane_get_caps {
	__u32 flags;
	__u32 size;
	__u32 abi_version;
	__u32 chip_family;
	__u32 section_size;
	__u32 bind_size;
	__u32 exec_io_size;
	__u32 pad;
};

/**
 * struct drm_ane_bo_init - DRM_IOCTL_ANE_BO_INIT argument
 * @handle: Out. Handle of the new buffer.
 * @pad: Must be zero.
 * @size: In. Size of the buffer in bytes, not zero.
 * @offset: Out. Offset to give to mmap() on the accel node.
 */
struct drm_ane_bo_init {
	__u32 handle;
	__u32 pad;
	__u64 size;
	__u64 offset;
};

/**
 * struct drm_ane_bo_free - DRM_IOCTL_ANE_BO_FREE argument
 * @handle: In. Handle from BO_INIT.
 * @pad: Must be zero.
 */
struct drm_ane_bo_free {
	__u32 handle;
	__u32 pad;
};

/**
 * struct drm_ane_submit - DRM_IOCTL_ANE_SUBMIT argument (ABI version 1)
 * @tsk_size: In. Size of the task microcode at the start of handles[0]: the
 *	@td_count task descriptors of @td_size bytes. Not zero, at least
 *	@td_count times @td_size, and smaller than handles[0].
 * @td_count: In. Number of task descriptors, 1 to 0xffff.
 * @td_size: In. Size of a task descriptor: a multiple of 4, 4 to 0x40000.
 * @handles: In. Buffer per tile: handles[0] holds the microcode and the
 *	weights, handles[1] must be 0 (the driver sets it from handles[0]),
 *	and 0 marks an unused tile.
 * @btsp_handle: In. Buffer with a copy of the first task descriptor, at
 *	least @td_size bytes. The engine starts from this copy.
 * @pad: Must be zero.
 *
 * A successful SUBMIT returns after the engine completed the work, with the
 * output buffers visible to the CPU.
 */
struct drm_ane_submit {
	__u64 tsk_size;
	__u32 td_count;
	__u32 td_size;
	__u32 handles[DRM_ANE_TILE_COUNT];
	__u32 btsp_handle;
	__u32 pad;
};

/**
 * struct drm_ane_section - one firmware program section (ABI version 2)
 * @id: Section id, 1 to %DRM_ANE_MAX_SECTIONS, unique in one PROG_LOAD.
 * @bo_handle: Buffer that holds the section bytes.
 * @size: Size of the section in bytes.
 * @offset: Offset of the section in the buffer.
 */
struct drm_ane_section {
	__u32 id;
	__u32 bo_handle;
	__u64 size;
	__u64 offset;
};

/**
 * struct drm_ane_generic_bind - buffer of a program (ABI version 2)
 * @buffer_id: Firmware buffer id.
 * @bo_handle: Buffer handle.
 * @type: Buffer type.
 * @pad: Must be zero.
 * @size: Size of the buffer in bytes.
 *
 * The driver checks @pad and ignores the other fields: the sections that
 * userspace builds already hold this information.
 */
struct drm_ane_generic_bind {
	__u32 buffer_id;
	__u32 bo_handle;
	__u32 type;
	__u32 pad;
	__u64 size;
};

/**
 * struct drm_ane_prog_load - DRM_IOCTL_ANE_PROG_LOAD argument (ABI version 2)
 * @sections_ptr: In. Pointer to struct drm_ane_section[@section_count].
 * @generic_ptr: In. Pointer to struct drm_ane_generic_bind[@generic_count].
 * @section_count: In. 1 to %DRM_ANE_MAX_SECTIONS.
 * @generic_count: In. 0 to %DRM_ANE_MAX_BINDS.
 * @prog_id_out: Out. Opaque program id. Identical sections in one firmware
 *	generation give the same program id.
 * @pad: Must be zero.
 *
 * Device reset invalidates program and process ids. Reload the sections and
 * create the process again before resubmitting.
 */
struct drm_ane_prog_load {
	__u64 sections_ptr;
	__u64 generic_ptr;
	__u32 section_count;
	__u32 generic_count;
	__u32 prog_id_out;
	__u32 pad;
};

/**
 * struct drm_ane_proc_create - DRM_IOCTL_ANE_PROC_CREATE argument
 *	(ABI version 2)
 * @prog_id: In. Program id from PROG_LOAD.
 * @proc_id_out: Out. Opaque process id.
 */
struct drm_ane_proc_create {
	__u32 prog_id;
	__u32 proc_id_out;
};

/**
 * struct drm_ane_exec_io - input or output buffer of one EXEC
 *	(ABI version 2)
 * @buffer_id: Firmware buffer id.
 * @bo_handle: Buffer handle. The driver gives its address to the firmware.
 * @type: Buffer type.
 * @flags: No flags are defined: must be zero.
 * @reserved: Must be zero.
 * @size: Size of the buffer in bytes.
 */
struct drm_ane_exec_io {
	__u32 buffer_id;
	__u32 bo_handle;
	__u32 type;
	__u32 flags;
	__u64 reserved;
	__u64 size;
};

/**
 * struct drm_ane_exec - DRM_IOCTL_ANE_EXEC argument (ABI version 2)
 * @prog_id: In. Program id from PROG_LOAD.
 * @proc_id: In. Process id from PROC_CREATE.
 * @priority: In. 2 to 7.
 * @timeout_ms: In. Firmware completion timeout, capped at 5000 ms; 0 uses
 *	5000 ms.
 * @count: In. 1 to %DRM_ANE_MAX_BINDS.
 * @pad: Must be zero.
 * @io_ptr: In. Pointer to struct drm_ane_exec_io[@count].
 *
 * A successful EXEC returns after the firmware acknowledged completion and
 * the task queue is idle, with the output buffers visible to the CPU.
 */
struct drm_ane_exec {
	__u32 prog_id;
	__u32 proc_id;
	__u32 priority;
	__u32 timeout_ms;
	__u32 count;
	__u32 pad;
	__u64 io_ptr;
};

/**
 * struct drm_ane_h16_section - one host-built program section (ABI version 3)
 * @kind: %DRM_ANE_H16_SECTION_GENERIC to
 *	%DRM_ANE_H16_SECTION_TEXT_PROPERTIES, each exactly once.
 * @size: Bytes at @data_ptr, %DRM_ANE_H16_SECTION_MIN to
 *	%DRM_ANE_H16_SECTION_MAX and a multiple of 4.
 * @data_ptr: Pointer to the section bytes. The kernel copies them.
 */
struct drm_ane_h16_section {
	__u32 kind;
	__u32 size;
	__u64 data_ptr;
};

/**
 * struct drm_ane_h16_load - DRM_IOCTL_ANE_H16_LOAD argument (ABI version 3)
 * @hwx_ptr: In. Pointer to the original HWX file.
 * @hwx_size: In. Size of the HWX file in bytes.
 * @sections_ptr: In. Pointer to struct drm_ane_h16_section[@section_count].
 * @section_count: In. Exactly %DRM_ANE_H16_SECTIONS.
 * @flags: In. No flags are defined: must be zero.
 * @surface_count: Out. Surfaces of the program, inputs and the one output in
 *	the HWX segment order that H16_EXEC uses. A program's __DATA scratch
 *	is kernel-owned and not a surface.
 * @pad: Must be zero.
 * @surface_size: Out. Bytes of each surface, for @surface_count entries.
 * @surface_direction: Out. 1 for an input, 2 for the output.
 *
 * The kernel validates the HWX, places every section in its own private
 * mapping, then loads the program and creates its process. The program
 * belongs to this file and is unloaded when the file closes. The HWX must
 * be compiled for the engine that GET_CAPS reports. On H17 a firmware boot
 * accepts any number of programs, one per file. On H16 a firmware boot
 * accepts one program: a second H16_LOAD fails with -EBUSY, also after the
 * first program closed, and a reboot starts a new firmware boot.
 */
struct drm_ane_h16_load {
	__u64 hwx_ptr;
	__u64 hwx_size;
	__u64 sections_ptr;
	__u32 section_count;
	__u32 flags;
	__u32 surface_count;
	__u32 pad;
	__u64 surface_size[DRM_ANE_H16_SURFACES];
	__u32 surface_direction[DRM_ANE_H16_SURFACES];
};

/**
 * struct drm_ane_h16_buffer - one surface of an H16_EXEC (ABI version 3)
 * @ptr: In. Pointer to the bytes: read for an input, written for the output.
 * @size: In. Must equal the surface size that H16_LOAD reported.
 */
struct drm_ane_h16_buffer {
	__u64 ptr;
	__u64 size;
};

/*
 * mmap() offset of surface @i of the loaded program (ABI version 3, accel
 * minor 1). A mapping starts at the surface's first byte and may cover a
 * prefix of it, rounded up to whole pages; read for any surface, write for
 * an input. The pages stay valid until the mapping is gone, also after the
 * program closed; they are never given to another program before that.
 */
#define DRM_ANE_H16_MMAP_OFFSET(i)	((__u64)((i) + 1) << 32)

/* H16_EXEC flags */
/*
 * Run on the surfaces in place: inputs are what the caller wrote through
 * its mappings, the output stays in its surface and is not poisoned
 * first. @buffers_ptr must be 0; @count is still the surface count.
 */
#define DRM_ANE_H16_EXEC_IN_PLACE	(1u << 0)

/**
 * struct drm_ane_h16_exec - DRM_IOCTL_ANE_H16_EXEC argument (ABI version 3)
 * @buffers_ptr: In. Pointer to struct drm_ane_h16_buffer[@count], one entry
 *	per surface in the order that H16_LOAD reported.
 * @count: In. The surface count of the loaded program.
 * @flags: In. 0, or %DRM_ANE_H16_EXEC_IN_PLACE.
 *
 * A successful H16_EXEC returns after the firmware reported the end of the
 * request, with the output copied to its buffer (or left in its surface for
 * %DRM_ANE_H16_EXEC_IN_PLACE). Programs, sections and scratch stay in
 * private kernel mappings. Without the flag inputs are copied in before the
 * request and the output is copied out after it; with it the caller reads
 * and writes the surfaces through its own mappings.
 */
struct drm_ane_h16_exec {
	__u64 buffers_ptr;
	__u32 count;
	__u32 flags;
};

#define DRM_IOCTL_ANE_GET_CAPS \
	DRM_IOWR(DRM_COMMAND_BASE + DRM_ANE_GET_CAPS, struct drm_ane_get_caps)
#define DRM_IOCTL_ANE_BO_INIT \
	DRM_IOWR(DRM_COMMAND_BASE + DRM_ANE_BO_INIT, struct drm_ane_bo_init)
#define DRM_IOCTL_ANE_BO_FREE \
	DRM_IOWR(DRM_COMMAND_BASE + DRM_ANE_BO_FREE, struct drm_ane_bo_free)
#define DRM_IOCTL_ANE_SUBMIT \
	DRM_IOWR(DRM_COMMAND_BASE + DRM_ANE_SUBMIT, struct drm_ane_submit)
#define DRM_IOCTL_ANE_PROG_LOAD \
	DRM_IOWR(DRM_COMMAND_BASE + DRM_ANE_PROG_LOAD, struct drm_ane_prog_load)
#define DRM_IOCTL_ANE_PROC_CREATE \
	DRM_IOWR(DRM_COMMAND_BASE + DRM_ANE_PROC_CREATE, struct drm_ane_proc_create)
#define DRM_IOCTL_ANE_EXEC \
	DRM_IOWR(DRM_COMMAND_BASE + DRM_ANE_EXEC, struct drm_ane_exec)
#define DRM_IOCTL_ANE_H16_LOAD \
	DRM_IOWR(DRM_COMMAND_BASE + DRM_ANE_H16_LOAD, struct drm_ane_h16_load)
#define DRM_IOCTL_ANE_H16_EXEC \
	DRM_IOW(DRM_COMMAND_BASE + DRM_ANE_H16_EXEC, struct drm_ane_h16_exec)

#if defined(__cplusplus)
}
#endif

#endif /* __DRM_UAPI_ANE_ACCEL_H__ */
