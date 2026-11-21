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
layout(set=0, binding=3, std430) readonly buffer _MlxBuffer3 { uint16_t data[]; } _b3;
layout(set=0, binding=4, std430) readonly buffer _MlxBuffer4 { uint16_t data[]; } _b4;
layout(set=0, binding=5, std430) readonly buffer _MlxBuffer5 { uint16_t data[]; } _b5;
layout(set=0, binding=6, std430) readonly buffer _MlxBuffer6 { uint16_t data[]; } _b6;
layout(set=0, binding=7, std430) readonly buffer _MlxBuffer7 { uint16_t data[]; } _b7;
layout(set=0, binding=8, std430) readonly buffer _MlxBuffer8 { float data[]; } _b8;
layout(set=0, binding=9, std430) readonly buffer _MlxBuffer9 { float data[]; } _b9;
layout(set=0, binding=10, std430) readonly buffer _MlxBuffer10 { float data[]; } _b10;
layout(set=0, binding=11, std430) buffer _MlxBuffer11 { uint16_t data[]; } _b11;
layout(set=0, binding=12, std430) buffer _MlxBuffer12 { uint16_t data[]; } _b12;
layout(set=0, binding=13, std430) buffer _MlxBuffer13 { uint16_t data[]; } _b13;
layout(set=0, binding=14, std430) buffer _MlxBuffer14 { float data[]; } _b14;
layout(set=0, binding=15, std430) buffer _MlxBuffer15 { uint16_t data[]; } _b15;
layout(set=0, binding=16, std430) buffer _MlxBuffer16 { uint16_t data[]; } _b16;
layout(set=0, binding=17, std430) buffer _MlxBuffer17 { uint16_t data[]; } _b17;
layout(set=0, binding=18, std430) buffer _MlxBuffer18 { uint16_t data[]; } _b18;
shared float tgq[4];
shared float tgk[4];
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

#define _mlx_arg8 _b8.data
#define _mlx_arg9 _b9.data
#define _mlx_arg10 _b10.data
#define _mlx_arg14 _b14.data
void main() {
if (gl_GlobalInvocationID.x >= 12288u || gl_GlobalInvocationID.y >= 1u || gl_GlobalInvocationID.z >= 1u) return;


    uint h = uint(gl_WorkGroupID.x);
    uint d = uint(gl_LocalInvocationID.x);
    uint c = uint(h * 128u + d);
               

    float taps_q[4] = { float(_mlx_bf16_to_float(_b2.data[c])), float(_mlx_bf16_to_float(_b2.data[12288u + c])), float(_mlx_bf16_to_float(_b2.data[24576u + c])), float(_mlx_bf16_to_float(_b0.data[c])) };
    float taps_k[4] = { float(_mlx_bf16_to_float(_b3.data[c])), float(_mlx_bf16_to_float(_b3.data[12288u + c])), float(_mlx_bf16_to_float(_b3.data[24576u + c])), float(_mlx_bf16_to_float(_b0.data[12288u + c])) };
    float taps_v[4] = { float(_mlx_bf16_to_float(_b4.data[c])), float(_mlx_bf16_to_float(_b4.data[12288u + c])), float(_mlx_bf16_to_float(_b4.data[24576u + c])), float(_mlx_bf16_to_float(_b0.data[24576u + c])) };

    float aq = 0.0f, ak = 0.0f, av = 0.0f;
    for (uint t = uint(0); t < 4; t++) {
        aq += taps_q[t] * float(_mlx_bf16_to_float(_b5.data[c * 4u + t]));
        ak += taps_k[t] * float(_mlx_bf16_to_float(_b6.data[c * 4u + t]));
        av += taps_v[t] * float(_mlx_bf16_to_float(_b7.data[c * 4u + t]));
    }
    float cq = float(aq), ck = float(ak), cv = float(av);      
    float sq_ = mlx_sigmoid(cq);
    float sk_ = mlx_sigmoid(ck);
    float sv_ = mlx_sigmoid(cv);
    float yq = ((float(cq) * float(sq_));            
    float yk = ((float(ck) * float(sk_));
    float yv = ((float(cv) * float(sv_));

    
    _b16.data[c] = _mlx_float_to_bf16(_mlx_bf16_to_float(_b2.data[12288u + c])); _b16.data[12288u + c] = _mlx_float_to_bf16(_mlx_bf16_to_float(_b2.data[24576u + c])); _b16.data[24576u + c] = _mlx_float_to_bf16(_mlx_bf16_to_float(_b0.data[c]));
    _b17.data[c] = _mlx_float_to_bf16(_mlx_bf16_to_float(_b3.data[12288u + c])); _b17.data[12288u + c] = _mlx_float_to_bf16(_mlx_bf16_to_float(_b3.data[24576u + c])); _b17.data[24576u + c] = _mlx_float_to_bf16(_mlx_bf16_to_float(_b0.data[12288u + c]));
    _b18.data[c] = _mlx_float_to_bf16(_mlx_bf16_to_float(_b4.data[12288u + c])); _b18.data[12288u + c] = _mlx_float_to_bf16(_mlx_bf16_to_float(_b4.data[24576u + c])); _b18.data[24576u + c] = _mlx_float_to_bf16(_mlx_bf16_to_float(_b0.data[24576u + c]));

    
    
    
    uint sg = uint(d >> 5);
    float pq = subgroupAdd(yq * yq);
    float pk = subgroupAdd(yk * yk);
    if ((d & 31u) == 0u) { tgq[sg] = pq; tgk[sg] = pk; }
    barrier();
    float mq = (tgq[0] + tgq[1] + tgq[2] + tgq[3]) / 128.0f + _mlx_arg10[(0) + (3)];
    float mk = (tgk[0] + tgk[1] + tgk[2] + tgk[3]) / 128.0f + _mlx_arg10[(0) + (3)];
    float nq = float((yq * inversesqrt(mq)));                   
    float nk = float((yk * inversesqrt(mk)));
    _b11.data[c] = _mlx_float_to_bf16(float((float * _mlx_arg10[(0) + (2)])));                            
    _b12.data[c] = _mlx_float_to_bf16(float((float * _mlx_arg10[(0) + (1)])));                            
    _b13.data[c] = _mlx_float_to_bf16(float(yv));

    
    
    
    float zf = float(_mlx_bf16_to_float(_b1.data[c])) + _mlx_arg8[c];
    float s = 1.0f / (1.0f + exp(-(exp(_mlx_arg9[d]) * zf)));
    _mlx_arg14[c] = float(exp(_mlx_arg10[(0) + (0)] * s));

    if (d == 0u) { _b15.data[h] = _mlx_float_to_bf16(mlx_sigmoid(_mlx_bf16_to_float(_b0.data[49280u + h]))); }


}
