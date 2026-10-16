# vkbw: needs a C compiler, the Vulkan headers and loader (-lvulkan), and glslangValidator.
CC ?= cc
GLSLANG ?= glslangValidator
CFLAGS ?= -O2 -Wall -Wextra
VKFLAGS := -V --target-env vulkan1.1 -S comp
LDLIBS ?= -lvulkan

vkbw: vkbw.c shaders.h
	$(CC) $(CFLAGS) -o $@ vkbw.c $(LDLIBS)

shaders.h: shaders/membw.comp shaders/init.comp
	$(GLSLANG) $(VKFLAGS) -DW=2 --vn spv_w2 -o $@.2 shaders/membw.comp
	$(GLSLANG) $(VKFLAGS) -DW=4 --vn spv_w4 -o $@.4 shaders/membw.comp
	$(GLSLANG) $(VKFLAGS) -DW=8 --vn spv_w8 -o $@.8 shaders/membw.comp
	$(GLSLANG) $(VKFLAGS) -DW=16 --vn spv_w16 -o $@.16 shaders/membw.comp
	$(GLSLANG) $(VKFLAGS) --vn spv_init -o $@.i shaders/init.comp
	cat $@.i $@.2 $@.4 $@.8 $@.16 > $@
	rm -f $@.i $@.2 $@.4 $@.8 $@.16

clean:
	rm -f vkbw shaders.h

.PHONY: clean
