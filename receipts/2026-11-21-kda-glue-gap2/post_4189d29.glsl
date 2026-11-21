#version 460
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_shuffle : require
#define __FAST_MATH__ 0
layout(local_size_x=128, local_size_y=1, local_size_z=1) in;
layout(set=0, binding=0, std430) readonly buffer _MlxBuffer0 { uint16_t data[]; } _b0;
layout(set=0, binding=1, std430) readonly buffer _MlxBuffer1 { uint16_t data[]; } _b1;
layout(set=0, binding=2, std430) readonly buffer _MlxBuffer2 { uint16_t data[]; } _b2;
layout(set=0, binding=3, std430) readonly buffer _MlxBuffer3 { float data[]; } _b3;
layout(set=0, binding=4, std430) buffer _MlxBuffer4 { uint16_t data[]; } _b4;
shared float tgs[4];
float _mlx_bf16_to_float(uint16_t value) { return uintBitsToFloat(uint(value) << 16); }
uint16_t _mlx_float_to_bf16(float value) { uint bits = floatBitsToUint(value); uint rounded = bits + 0x7fffu + ((bits >> 16) & 1u); return uint16_t(rounded >> 16); }
float _mlx_bf16_round(float value) { return _mlx_bf16_to_float(_mlx_float_to_bf16(value)); }
float _mlx_bf16_round_trip(float value) { return _mlx_bf16_round(value); }

 float mlx_sigmoid(float x) {
    float e = float(exp(abs(float(x))));
    float den = float((1.0f + float(e)));
    float y = float((1.0f / float(den)));
    return (float(x) < 0.0f) ? y : float((1.0f - float(y)));
}


#define _mlx_arg3 _b3.data
void main() {
if (gl_GlobalInvocationID.x >= 12288u || gl_GlobalInvocationID.y >= 1u || gl_GlobalInvocationID.z >= 1u) return;


    uint h = uint(gl_WorkGroupID.x);
    uint d = uint(gl_LocalInvocationID.x);
    uint c = uint(h * 128u + d);
               

    float x = float(_mlx_bf16_to_float(_b0.data[c]));
    
    uint sg = uint(d >> 5);
    float p = subgroupAdd(x * x);
    if ((d & 31u) == 0u) { tgs[sg] = p; }
    barrier();
    float m = (tgs[0] + tgs[1] + tgs[2] + tgs[3]) / 128.0f + _mlx_arg3[(0) + (0)];
    
    float nx0 = float((x * inversesqrt(m)));
    float nx = float((float(nx0) * float(_mlx_bf16_to_float(_b2.data[d]))));
    float sg_ = mlx_sigmoid(_mlx_bf16_to_float(_b1.data[36864u + c]));
    _b4.data[c] = _mlx_float_to_bf16(float((float(nx) * float(sg_))));


}
