#pragma once

#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>
#include <numeric>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

// memory mapping on Linux/macOS
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

namespace qwen_loader
{

using bf16 = __nv_bfloat16;
const int N_LAYERS = 28;

struct
AttentionWeights
{
    bf16* q_proj_weight;
    bf16* k_proj_weight;
    bf16* v_proj_weight;
    bf16* o_proj_weight;
    bf16* q_norm_weight;
    bf16* k_norm_weight;
};

struct
FFNWeights
{
    bf16* gate_proj_weight;
    bf16* up_proj_weight;
    bf16* down_proj_weight;
};

struct
TransformerBlockWeights
{
    bf16* input_layernorm_weight;
    bf16* post_attention_layernorm_weight;
    AttentionWeights attention;
    FFNWeights ffn;
};

struct 
QwenWeights
{
    bf16* token_embedding_table; // model.embed_tokens.weight
    TransformerBlockWeights layers[N_LAYERS];
    bf16* final_norm_weight;     // model.norm.weight
    bf16* output_head_weight;    // lm_head.weight
    void* _gpu_mem_block = nullptr;

    ~QwenWeights()
    {
        if (_gpu_mem_block) { CUDA_CHECK(cudaFree(_gpu_mem_block)); }
    }
};

// total model size: 1433.62 MB
const size_t TOTAL_WEIGHTS_BYTES = 1503264768;
const int NUM_TENSORS = 311;

struct TensorMetadata { const char* name; size_t offset; size_t size_bytes; };

const TensorMetadata 
TENSOR_METADATA[NUM_TENSORS] =
{
    {"lm_head.weight", 0, 311164928},
    {"model.embed_tokens.weight", 311164928, 311164928},
    {"model.layers.0.input_layernorm.weight", 622329856, 2048},
    {"model.layers.0.mlp.down_proj.weight", 622331904, 6291456},
    {"model.layers.0.mlp.gate_proj.weight", 628623360, 6291456},
    {"model.layers.0.mlp.up_proj.weight", 634914816, 6291456},
    {"model.layers.0.post_attention_layernorm.weight", 641206272, 2048},
    {"model.layers.0.self_attn.k_norm.weight", 641208320, 256},
    {"model.layers.0.self_attn.k_proj.weight", 641208576, 2097152},
    {"model.layers.0.self_attn.o_proj.weight", 643305728, 4194304},
    {"model.layers.0.self_attn.q_norm.weight", 647500032, 256},
    {"model.layers.0.self_attn.q_proj.weight", 647500288, 4194304},
    {"model.layers.0.self_attn.v_proj.weight", 651694592, 2097152},
    {"model.layers.1.input_layernorm.weight", 653791744, 2048},
    {"model.layers.1.mlp.down_proj.weight", 653793792, 6291456},
    {"model.layers.1.mlp.gate_proj.weight", 660085248, 6291456},
    {"model.layers.1.mlp.up_proj.weight", 666376704, 6291456},
    {"model.layers.1.post_attention_layernorm.weight", 672668160, 2048},
    {"model.layers.1.self_attn.k_norm.weight", 672670208, 256},
    {"model.layers.1.self_attn.k_proj.weight", 672670464, 2097152},
    {"model.layers.1.self_attn.o_proj.weight", 674767616, 4194304},
    {"model.layers.1.self_attn.q_norm.weight", 678961920, 256},
    {"model.layers.1.self_attn.q_proj.weight", 678962176, 4194304},
    {"model.layers.1.self_attn.v_proj.weight", 683156480, 2097152},
    {"model.layers.10.input_layernorm.weight", 685253632, 2048},
    {"model.layers.10.mlp.down_proj.weight", 685255680, 6291456},
    {"model.layers.10.mlp.gate_proj.weight", 691547136, 6291456},
    {"model.layers.10.mlp.up_proj.weight", 697838592, 6291456},
    {"model.layers.10.post_attention_layernorm.weight", 704130048, 2048},
    {"model.layers.10.self_attn.k_norm.weight", 704132096, 256},
    {"model.layers.10.self_attn.k_proj.weight", 704132352, 2097152},
    {"model.layers.10.self_attn.o_proj.weight", 706229504, 4194304},
    {"model.layers.10.self_attn.q_norm.weight", 710423808, 256},
    {"model.layers.10.self_attn.q_proj.weight", 710424064, 4194304},
    {"model.layers.10.self_attn.v_proj.weight", 714618368, 2097152},
    {"model.layers.11.input_layernorm.weight", 716715520, 2048},
    {"model.layers.11.mlp.down_proj.weight", 716717568, 6291456},
    {"model.layers.11.mlp.gate_proj.weight", 723009024, 6291456},
    {"model.layers.11.mlp.up_proj.weight", 729300480, 6291456},
    {"model.layers.11.post_attention_layernorm.weight", 735591936, 2048},
    {"model.layers.11.self_attn.k_norm.weight", 735593984, 256},
    {"model.layers.11.self_attn.k_proj.weight", 735594240, 2097152},
    {"model.layers.11.self_attn.o_proj.weight", 737691392, 4194304},
    {"model.layers.11.self_attn.q_norm.weight", 741885696, 256},
    {"model.layers.11.self_attn.q_proj.weight", 741885952, 4194304},
    {"model.layers.11.self_attn.v_proj.weight", 746080256, 2097152},
    {"model.layers.12.input_layernorm.weight", 748177408, 2048},
    {"model.layers.12.mlp.down_proj.weight", 748179456, 6291456},
    {"model.layers.12.mlp.gate_proj.weight", 754470912, 6291456},
    {"model.layers.12.mlp.up_proj.weight", 760762368, 6291456},
    {"model.layers.12.post_attention_layernorm.weight", 767053824, 2048},
    {"model.layers.12.self_attn.k_norm.weight", 767055872, 256},
    {"model.layers.12.self_attn.k_proj.weight", 767056128, 2097152},
    {"model.layers.12.self_attn.o_proj.weight", 769153280, 4194304},
    {"model.layers.12.self_attn.q_norm.weight", 773347584, 256},
    {"model.layers.12.self_attn.q_proj.weight", 773347840, 4194304},
    {"model.layers.12.self_attn.v_proj.weight", 777542144, 2097152},
    {"model.layers.13.input_layernorm.weight", 779639296, 2048},
    {"model.layers.13.mlp.down_proj.weight", 779641344, 6291456},
    {"model.layers.13.mlp.gate_proj.weight", 785932800, 6291456},
    {"model.layers.13.mlp.up_proj.weight", 792224256, 6291456},
    {"model.layers.13.post_attention_layernorm.weight", 798515712, 2048},
    {"model.layers.13.self_attn.k_norm.weight", 798517760, 256},
    {"model.layers.13.self_attn.k_proj.weight", 798518016, 2097152},
    {"model.layers.13.self_attn.o_proj.weight", 800615168, 4194304},
    {"model.layers.13.self_attn.q_norm.weight", 804809472, 256},
    {"model.layers.13.self_attn.q_proj.weight", 804809728, 4194304},
    {"model.layers.13.self_attn.v_proj.weight", 809004032, 2097152},
    {"model.layers.14.input_layernorm.weight", 811101184, 2048},
    {"model.layers.14.mlp.down_proj.weight", 811103232, 6291456},
    {"model.layers.14.mlp.gate_proj.weight", 817394688, 6291456},
    {"model.layers.14.mlp.up_proj.weight", 823686144, 6291456},
    {"model.layers.14.post_attention_layernorm.weight", 829977600, 2048},
    {"model.layers.14.self_attn.k_norm.weight", 829979648, 256},
    {"model.layers.14.self_attn.k_proj.weight", 829979904, 2097152},
    {"model.layers.14.self_attn.o_proj.weight", 832077056, 4194304},
    {"model.layers.14.self_attn.q_norm.weight", 836271360, 256},
    {"model.layers.14.self_attn.q_proj.weight", 836271616, 4194304},
    {"model.layers.14.self_attn.v_proj.weight", 840465920, 2097152},
    {"model.layers.15.input_layernorm.weight", 842563072, 2048},
    {"model.layers.15.mlp.down_proj.weight", 842565120, 6291456},
    {"model.layers.15.mlp.gate_proj.weight", 848856576, 6291456},
    {"model.layers.15.mlp.up_proj.weight", 855148032, 6291456},
    {"model.layers.15.post_attention_layernorm.weight", 861439488, 2048},
    {"model.layers.15.self_attn.k_norm.weight", 861441536, 256},
    {"model.layers.15.self_attn.k_proj.weight", 861441792, 2097152},
    {"model.layers.15.self_attn.o_proj.weight", 863538944, 4194304},
    {"model.layers.15.self_attn.q_norm.weight", 867733248, 256},
    {"model.layers.15.self_attn.q_proj.weight", 867733504, 4194304},
    {"model.layers.15.self_attn.v_proj.weight", 871927808, 2097152},
    {"model.layers.16.input_layernorm.weight", 874024960, 2048},
    {"model.layers.16.mlp.down_proj.weight", 874027008, 6291456},
    {"model.layers.16.mlp.gate_proj.weight", 880318464, 6291456},
    {"model.layers.16.mlp.up_proj.weight", 886609920, 6291456},
    {"model.layers.16.post_attention_layernorm.weight", 892901376, 2048},
    {"model.layers.16.self_attn.k_norm.weight", 892903424, 256},
    {"model.layers.16.self_attn.k_proj.weight", 892903680, 2097152},
    {"model.layers.16.self_attn.o_proj.weight", 895000832, 4194304},
    {"model.layers.16.self_attn.q_norm.weight", 899195136, 256},
    {"model.layers.16.self_attn.q_proj.weight", 899195392, 4194304},
    {"model.layers.16.self_attn.v_proj.weight", 903389696, 2097152},
    {"model.layers.17.input_layernorm.weight", 905486848, 2048},
    {"model.layers.17.mlp.down_proj.weight", 905488896, 6291456},
    {"model.layers.17.mlp.gate_proj.weight", 911780352, 6291456},
    {"model.layers.17.mlp.up_proj.weight", 918071808, 6291456},
    {"model.layers.17.post_attention_layernorm.weight", 924363264, 2048},
    {"model.layers.17.self_attn.k_norm.weight", 924365312, 256},
    {"model.layers.17.self_attn.k_proj.weight", 924365568, 2097152},
    {"model.layers.17.self_attn.o_proj.weight", 926462720, 4194304},
    {"model.layers.17.self_attn.q_norm.weight", 930657024, 256},
    {"model.layers.17.self_attn.q_proj.weight", 930657280, 4194304},
    {"model.layers.17.self_attn.v_proj.weight", 934851584, 2097152},
    {"model.layers.18.input_layernorm.weight", 936948736, 2048},
    {"model.layers.18.mlp.down_proj.weight", 936950784, 6291456},
    {"model.layers.18.mlp.gate_proj.weight", 943242240, 6291456},
    {"model.layers.18.mlp.up_proj.weight", 949533696, 6291456},
    {"model.layers.18.post_attention_layernorm.weight", 955825152, 2048},
    {"model.layers.18.self_attn.k_norm.weight", 955827200, 256},
    {"model.layers.18.self_attn.k_proj.weight", 955827456, 2097152},
    {"model.layers.18.self_attn.o_proj.weight", 957924608, 4194304},
    {"model.layers.18.self_attn.q_norm.weight", 962118912, 256},
    {"model.layers.18.self_attn.q_proj.weight", 962119168, 4194304},
    {"model.layers.18.self_attn.v_proj.weight", 966313472, 2097152},
    {"model.layers.19.input_layernorm.weight", 968410624, 2048},
    {"model.layers.19.mlp.down_proj.weight", 968412672, 6291456},
    {"model.layers.19.mlp.gate_proj.weight", 974704128, 6291456},
    {"model.layers.19.mlp.up_proj.weight", 980995584, 6291456},
    {"model.layers.19.post_attention_layernorm.weight", 987287040, 2048},
    {"model.layers.19.self_attn.k_norm.weight", 987289088, 256},
    {"model.layers.19.self_attn.k_proj.weight", 987289344, 2097152},
    {"model.layers.19.self_attn.o_proj.weight", 989386496, 4194304},
    {"model.layers.19.self_attn.q_norm.weight", 993580800, 256},
    {"model.layers.19.self_attn.q_proj.weight", 993581056, 4194304},
    {"model.layers.19.self_attn.v_proj.weight", 997775360, 2097152},
    {"model.layers.2.input_layernorm.weight", 999872512, 2048},
    {"model.layers.2.mlp.down_proj.weight", 999874560, 6291456},
    {"model.layers.2.mlp.gate_proj.weight", 1006166016, 6291456},
    {"model.layers.2.mlp.up_proj.weight", 1012457472, 6291456},
    {"model.layers.2.post_attention_layernorm.weight", 1018748928, 2048},
    {"model.layers.2.self_attn.k_norm.weight", 1018750976, 256},
    {"model.layers.2.self_attn.k_proj.weight", 1018751232, 2097152},
    {"model.layers.2.self_attn.o_proj.weight", 1020848384, 4194304},
    {"model.layers.2.self_attn.q_norm.weight", 1025042688, 256},
    {"model.layers.2.self_attn.q_proj.weight", 1025042944, 4194304},
    {"model.layers.2.self_attn.v_proj.weight", 1029237248, 2097152},
    {"model.layers.20.input_layernorm.weight", 1031334400, 2048},
    {"model.layers.20.mlp.down_proj.weight", 1031336448, 6291456},
    {"model.layers.20.mlp.gate_proj.weight", 1037627904, 6291456},
    {"model.layers.20.mlp.up_proj.weight", 1043919360, 6291456},
    {"model.layers.20.post_attention_layernorm.weight", 1050210816, 2048},
    {"model.layers.20.self_attn.k_norm.weight", 1050212864, 256},
    {"model.layers.20.self_attn.k_proj.weight", 1050213120, 2097152},
    {"model.layers.20.self_attn.o_proj.weight", 1052310272, 4194304},
    {"model.layers.20.self_attn.q_norm.weight", 1056504576, 256},
    {"model.layers.20.self_attn.q_proj.weight", 1056504832, 4194304},
    {"model.layers.20.self_attn.v_proj.weight", 1060699136, 2097152},
    {"model.layers.21.input_layernorm.weight", 1062796288, 2048},
    {"model.layers.21.mlp.down_proj.weight", 1062798336, 6291456},
    {"model.layers.21.mlp.gate_proj.weight", 1069089792, 6291456},
    {"model.layers.21.mlp.up_proj.weight", 1075381248, 6291456},
    {"model.layers.21.post_attention_layernorm.weight", 1081672704, 2048},
    {"model.layers.21.self_attn.k_norm.weight", 1081674752, 256},
    {"model.layers.21.self_attn.k_proj.weight", 1081675008, 2097152},
    {"model.layers.21.self_attn.o_proj.weight", 1083772160, 4194304},
    {"model.layers.21.self_attn.q_norm.weight", 1087966464, 256},
    {"model.layers.21.self_attn.q_proj.weight", 1087966720, 4194304},
    {"model.layers.21.self_attn.v_proj.weight", 1092161024, 2097152},
    {"model.layers.22.input_layernorm.weight", 1094258176, 2048},
    {"model.layers.22.mlp.down_proj.weight", 1094260224, 6291456},
    {"model.layers.22.mlp.gate_proj.weight", 1100551680, 6291456},
    {"model.layers.22.mlp.up_proj.weight", 1106843136, 6291456},
    {"model.layers.22.post_attention_layernorm.weight", 1113134592, 2048},
    {"model.layers.22.self_attn.k_norm.weight", 1113136640, 256},
    {"model.layers.22.self_attn.k_proj.weight", 1113136896, 2097152},
    {"model.layers.22.self_attn.o_proj.weight", 1115234048, 4194304},
    {"model.layers.22.self_attn.q_norm.weight", 1119428352, 256},
    {"model.layers.22.self_attn.q_proj.weight", 1119428608, 4194304},
    {"model.layers.22.self_attn.v_proj.weight", 1123622912, 2097152},
    {"model.layers.23.input_layernorm.weight", 1125720064, 2048},
    {"model.layers.23.mlp.down_proj.weight", 1125722112, 6291456},
    {"model.layers.23.mlp.gate_proj.weight", 1132013568, 6291456},
    {"model.layers.23.mlp.up_proj.weight", 1138305024, 6291456},
    {"model.layers.23.post_attention_layernorm.weight", 1144596480, 2048},
    {"model.layers.23.self_attn.k_norm.weight", 1144598528, 256},
    {"model.layers.23.self_attn.k_proj.weight", 1144598784, 2097152},
    {"model.layers.23.self_attn.o_proj.weight", 1146695936, 4194304},
    {"model.layers.23.self_attn.q_norm.weight", 1150890240, 256},
    {"model.layers.23.self_attn.q_proj.weight", 1150890496, 4194304},
    {"model.layers.23.self_attn.v_proj.weight", 1155084800, 2097152},
    {"model.layers.24.input_layernorm.weight", 1157181952, 2048},
    {"model.layers.24.mlp.down_proj.weight", 1157184000, 6291456},
    {"model.layers.24.mlp.gate_proj.weight", 1163475456, 6291456},
    {"model.layers.24.mlp.up_proj.weight", 1169766912, 6291456},
    {"model.layers.24.post_attention_layernorm.weight", 1176058368, 2048},
    {"model.layers.24.self_attn.k_norm.weight", 1176060416, 256},
    {"model.layers.24.self_attn.k_proj.weight", 1176060672, 2097152},
    {"model.layers.24.self_attn.o_proj.weight", 1178157824, 4194304},
    {"model.layers.24.self_attn.q_norm.weight", 1182352128, 256},
    {"model.layers.24.self_attn.q_proj.weight", 1182352384, 4194304},
    {"model.layers.24.self_attn.v_proj.weight", 1186546688, 2097152},
    {"model.layers.25.input_layernorm.weight", 1188643840, 2048},
    {"model.layers.25.mlp.down_proj.weight", 1188645888, 6291456},
    {"model.layers.25.mlp.gate_proj.weight", 1194937344, 6291456},
    {"model.layers.25.mlp.up_proj.weight", 1201228800, 6291456},
    {"model.layers.25.post_attention_layernorm.weight", 1207520256, 2048},
    {"model.layers.25.self_attn.k_norm.weight", 1207522304, 256},
    {"model.layers.25.self_attn.k_proj.weight", 1207522560, 2097152},
    {"model.layers.25.self_attn.o_proj.weight", 1209619712, 4194304},
    {"model.layers.25.self_attn.q_norm.weight", 1213814016, 256},
    {"model.layers.25.self_attn.q_proj.weight", 1213814272, 4194304},
    {"model.layers.25.self_attn.v_proj.weight", 1218008576, 2097152},
    {"model.layers.26.input_layernorm.weight", 1220105728, 2048},
    {"model.layers.26.mlp.down_proj.weight", 1220107776, 6291456},
    {"model.layers.26.mlp.gate_proj.weight", 1226399232, 6291456},
    {"model.layers.26.mlp.up_proj.weight", 1232690688, 6291456},
    {"model.layers.26.post_attention_layernorm.weight", 1238982144, 2048},
    {"model.layers.26.self_attn.k_norm.weight", 1238984192, 256},
    {"model.layers.26.self_attn.k_proj.weight", 1238984448, 2097152},
    {"model.layers.26.self_attn.o_proj.weight", 1241081600, 4194304},
    {"model.layers.26.self_attn.q_norm.weight", 1245275904, 256},
    {"model.layers.26.self_attn.q_proj.weight", 1245276160, 4194304},
    {"model.layers.26.self_attn.v_proj.weight", 1249470464, 2097152},
    {"model.layers.27.input_layernorm.weight", 1251567616, 2048},
    {"model.layers.27.mlp.down_proj.weight", 1251569664, 6291456},
    {"model.layers.27.mlp.gate_proj.weight", 1257861120, 6291456},
    {"model.layers.27.mlp.up_proj.weight", 1264152576, 6291456},
    {"model.layers.27.post_attention_layernorm.weight", 1270444032, 2048},
    {"model.layers.27.self_attn.k_norm.weight", 1270446080, 256},
    {"model.layers.27.self_attn.k_proj.weight", 1270446336, 2097152},
    {"model.layers.27.self_attn.o_proj.weight", 1272543488, 4194304},
    {"model.layers.27.self_attn.q_norm.weight", 1276737792, 256},
    {"model.layers.27.self_attn.q_proj.weight", 1276738048, 4194304},
    {"model.layers.27.self_attn.v_proj.weight", 1280932352, 2097152},
    {"model.layers.3.input_layernorm.weight", 1283029504, 2048},
    {"model.layers.3.mlp.down_proj.weight", 1283031552, 6291456},
    {"model.layers.3.mlp.gate_proj.weight", 1289323008, 6291456},
    {"model.layers.3.mlp.up_proj.weight", 1295614464, 6291456},
    {"model.layers.3.post_attention_layernorm.weight", 1301905920, 2048},
    {"model.layers.3.self_attn.k_norm.weight", 1301907968, 256},
    {"model.layers.3.self_attn.k_proj.weight", 1301908224, 2097152},
    {"model.layers.3.self_attn.o_proj.weight", 1304005376, 4194304},
    {"model.layers.3.self_attn.q_norm.weight", 1308199680, 256},
    {"model.layers.3.self_attn.q_proj.weight", 1308199936, 4194304},
    {"model.layers.3.self_attn.v_proj.weight", 1312394240, 2097152},
    {"model.layers.4.input_layernorm.weight", 1314491392, 2048},
    {"model.layers.4.mlp.down_proj.weight", 1314493440, 6291456},
    {"model.layers.4.mlp.gate_proj.weight", 1320784896, 6291456},
    {"model.layers.4.mlp.up_proj.weight", 1327076352, 6291456},
    {"model.layers.4.post_attention_layernorm.weight", 1333367808, 2048},
    {"model.layers.4.self_attn.k_norm.weight", 1333369856, 256},
    {"model.layers.4.self_attn.k_proj.weight", 1333370112, 2097152},
    {"model.layers.4.self_attn.o_proj.weight", 1335467264, 4194304},
    {"model.layers.4.self_attn.q_norm.weight", 1339661568, 256},
    {"model.layers.4.self_attn.q_proj.weight", 1339661824, 4194304},
    {"model.layers.4.self_attn.v_proj.weight", 1343856128, 2097152},
    {"model.layers.5.input_layernorm.weight", 1345953280, 2048},
    {"model.layers.5.mlp.down_proj.weight", 1345955328, 6291456},
    {"model.layers.5.mlp.gate_proj.weight", 1352246784, 6291456},
    {"model.layers.5.mlp.up_proj.weight", 1358538240, 6291456},
    {"model.layers.5.post_attention_layernorm.weight", 1364829696, 2048},
    {"model.layers.5.self_attn.k_norm.weight", 1364831744, 256},
    {"model.layers.5.self_attn.k_proj.weight", 1364832000, 2097152},
    {"model.layers.5.self_attn.o_proj.weight", 1366929152, 4194304},
    {"model.layers.5.self_attn.q_norm.weight", 1371123456, 256},
    {"model.layers.5.self_attn.q_proj.weight", 1371123712, 4194304},
    {"model.layers.5.self_attn.v_proj.weight", 1375318016, 2097152},
    {"model.layers.6.input_layernorm.weight", 1377415168, 2048},
    {"model.layers.6.mlp.down_proj.weight", 1377417216, 6291456},
    {"model.layers.6.mlp.gate_proj.weight", 1383708672, 6291456},
    {"model.layers.6.mlp.up_proj.weight", 1390000128, 6291456},
    {"model.layers.6.post_attention_layernorm.weight", 1396291584, 2048},
    {"model.layers.6.self_attn.k_norm.weight", 1396293632, 256},
    {"model.layers.6.self_attn.k_proj.weight", 1396293888, 2097152},
    {"model.layers.6.self_attn.o_proj.weight", 1398391040, 4194304},
    {"model.layers.6.self_attn.q_norm.weight", 1402585344, 256},
    {"model.layers.6.self_attn.q_proj.weight", 1402585600, 4194304},
    {"model.layers.6.self_attn.v_proj.weight", 1406779904, 2097152},
    {"model.layers.7.input_layernorm.weight", 1408877056, 2048},
    {"model.layers.7.mlp.down_proj.weight", 1408879104, 6291456},
    {"model.layers.7.mlp.gate_proj.weight", 1415170560, 6291456},
    {"model.layers.7.mlp.up_proj.weight", 1421462016, 6291456},
    {"model.layers.7.post_attention_layernorm.weight", 1427753472, 2048},
    {"model.layers.7.self_attn.k_norm.weight", 1427755520, 256},
    {"model.layers.7.self_attn.k_proj.weight", 1427755776, 2097152},
    {"model.layers.7.self_attn.o_proj.weight", 1429852928, 4194304},
    {"model.layers.7.self_attn.q_norm.weight", 1434047232, 256},
    {"model.layers.7.self_attn.q_proj.weight", 1434047488, 4194304},
    {"model.layers.7.self_attn.v_proj.weight", 1438241792, 2097152},
    {"model.layers.8.input_layernorm.weight", 1440338944, 2048},
    {"model.layers.8.mlp.down_proj.weight", 1440340992, 6291456},
    {"model.layers.8.mlp.gate_proj.weight", 1446632448, 6291456},
    {"model.layers.8.mlp.up_proj.weight", 1452923904, 6291456},
    {"model.layers.8.post_attention_layernorm.weight", 1459215360, 2048},
    {"model.layers.8.self_attn.k_norm.weight", 1459217408, 256},
    {"model.layers.8.self_attn.k_proj.weight", 1459217664, 2097152},
    {"model.layers.8.self_attn.o_proj.weight", 1461314816, 4194304},
    {"model.layers.8.self_attn.q_norm.weight", 1465509120, 256},
    {"model.layers.8.self_attn.q_proj.weight", 1465509376, 4194304},
    {"model.layers.8.self_attn.v_proj.weight", 1469703680, 2097152},
    {"model.layers.9.input_layernorm.weight", 1471800832, 2048},
    {"model.layers.9.mlp.down_proj.weight", 1471802880, 6291456},
    {"model.layers.9.mlp.gate_proj.weight", 1478094336, 6291456},
    {"model.layers.9.mlp.up_proj.weight", 1484385792, 6291456},
    {"model.layers.9.post_attention_layernorm.weight", 1490677248, 2048},
    {"model.layers.9.self_attn.k_norm.weight", 1490679296, 256},
    {"model.layers.9.self_attn.k_proj.weight", 1490679552, 2097152},
    {"model.layers.9.self_attn.o_proj.weight", 1492776704, 4194304},
    {"model.layers.9.self_attn.q_norm.weight", 1496971008, 256},
    {"model.layers.9.self_attn.q_proj.weight", 1496971264, 4194304},
    {"model.layers.9.self_attn.v_proj.weight", 1501165568, 2097152},
    {"model.norm.weight", 1503262720, 2048},
};

void 
load_qwen_weights(
    const std::string& filepath, 
    QwenWeights& weights)
{
    // std::cout << "loading qwen weights from: " << filepath << std::endl;

    // step 1: Memory map the safetensors file
    int fd = open(filepath.c_str(), O_RDONLY);
    if (fd == -1) { throw std::runtime_error("Failed to open file: " + filepath); }

    struct stat file_stat;
    if (fstat(fd, &file_stat) == -1)
    {
        close(fd);
        throw std::runtime_error("Failed to get file stats.");
    }
    size_t file_size = file_stat.st_size;

    uint64_t json_header_len;
    if (pread(fd, &json_header_len, 8, 0) != 8)
    {
        close(fd);
        throw std::runtime_error("Failed to read safetensors header length.");
    }
    const size_t data_start_offset = 8 + json_header_len;

    char* mapped_file = (char*)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped_file == MAP_FAILED)
    {
        close(fd);
        throw std::runtime_error("Failed to mmap file.");
    }
    close(fd);

    char* data_ptr = mapped_file + data_start_offset;

    // step 2: allocate one giant block on the GPU
    CUDA_CHECK(cudaMalloc(&weights._gpu_mem_block, TOTAL_WEIGHTS_BYTES));
    char* all_weights_gpu = (char*)weights._gpu_mem_block;
    
    // step 3: copy each tensor from mmap'd memory to the GPU
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    // std::cout << "starting " << NUM_TENSORS << " async copies to GPU..." << std::endl;
    for (int i = 0; i < NUM_TENSORS; ++i)
    {
        const auto& meta = TENSOR_METADATA[i];
        void* gpu_target_ptr = all_weights_gpu + meta.offset;
        char* host_source_ptr = data_ptr + meta.offset;
        CUDA_CHECK(cudaMemcpyAsync(gpu_target_ptr, host_source_ptr, meta.size_bytes, cudaMemcpyHostToDevice, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));

    munmap(mapped_file, file_size);
    // std::cout << "all weights copied to GPU." << std::endl;

    // step 4: assign the pointers in our C++ struct (zero-cost)
    
    // std::cout << "assigning GPU pointers..." << std::endl;
    weights.output_head_weight    = (bf16*)(all_weights_gpu + TENSOR_METADATA[0].offset);
    weights.token_embedding_table = (bf16*)(all_weights_gpu + TENSOR_METADATA[1].offset);
    weights.final_norm_weight     = (bf16*)(all_weights_gpu + TENSOR_METADATA[310].offset);

    // safetensors file does not store layers sequentially
    const int layer_map[] = {0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24, 25, 26, 27, 3, 4, 5, 6, 7, 8, 9};
    const int tensors_per_layer = 11;
    
    // layer weights start at index 2 in the TENSOR_METADATA array
    int current_tensor_idx_offset = 2;

    for (int i = 0; i < N_LAYERS; ++i)
    {
        int layer_idx = layer_map[i];
        int start_idx = current_tensor_idx_offset + i * tensors_per_layer;
        
        TransformerBlockWeights& layer = weights.layers[layer_idx];
        layer.input_layernorm_weight          = (bf16*)(all_weights_gpu + TENSOR_METADATA[start_idx + 0].offset);
        layer.ffn.down_proj_weight            = (bf16*)(all_weights_gpu + TENSOR_METADATA[start_idx + 1].offset);
        layer.ffn.gate_proj_weight            = (bf16*)(all_weights_gpu + TENSOR_METADATA[start_idx + 2].offset);
        layer.ffn.up_proj_weight              = (bf16*)(all_weights_gpu + TENSOR_METADATA[start_idx + 3].offset);
        layer.post_attention_layernorm_weight = (bf16*)(all_weights_gpu + TENSOR_METADATA[start_idx + 4].offset);
        layer.attention.k_norm_weight         = (bf16*)(all_weights_gpu + TENSOR_METADATA[start_idx + 5].offset);
        layer.attention.k_proj_weight         = (bf16*)(all_weights_gpu + TENSOR_METADATA[start_idx + 6].offset);
        layer.attention.o_proj_weight         = (bf16*)(all_weights_gpu + TENSOR_METADATA[start_idx + 7].offset);
        layer.attention.q_norm_weight         = (bf16*)(all_weights_gpu + TENSOR_METADATA[start_idx + 8].offset);
        layer.attention.q_proj_weight         = (bf16*)(all_weights_gpu + TENSOR_METADATA[start_idx + 9].offset);
        layer.attention.v_proj_weight         = (bf16*)(all_weights_gpu + TENSOR_METADATA[start_idx + 10].offset);
    }
    
    // std::cout << "successfully loaded all weights to GPU." << std::endl;
}

} // namespace qwen_loader
