#pragma once

#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "internal/metal.h"

struct mindfour_tokenizer
{
    atomic_uint_least64_t ref_count;

    void *mapping_ptr;
    size_t mapping_size;

    const char *regex_ptr;
    const char *tokens_ptr;

    uint32_t num_text_tokens;
    uint32_t num_special_tokens;

    uint32_t special_token_id[mindfour_special_token_max - 1];
};

struct mindfour_model
{
    atomic_uint_least64_t ref_count;

    struct mindfour_tokenizer *tokenizer;

    void *mapping_ptr;
    size_t mapping_size;

    uint32_t context_length;
    uint32_t num_blocks;
    uint32_t num_experts;
    uint32_t num_active_experts;
    uint32_t embedding_dim;
    uint32_t mlp_dim;
    float swiglu_limit;
    uint32_t head_dim;
    uint32_t num_heads;
    uint32_t num_kv_heads;
    uint32_t attention_window;
    float rope_theta;
    float interpolation_scale;
    float yarn_offset;
    float yarn_scale;
    float yarn_multiplier;
    float rmsnorm_epsilon;

    uint32_t vocabulary_size;

    size_t max_batch_tokens;

    bool lock_memory;

    size_t weights_size;
    size_t allocation_size;

    struct mindfour_metal_device device;
    size_t max_threadgroups;
    struct mindfour_metal_command_queue command_queue;
    struct mindfour_metal_library library;
    struct mindfour_metal_function bf16_f32_embeddings_fn;
    struct mindfour_metal_function f32_bf16w_rmsnorm_fn;
    struct mindfour_metal_function f32_bf16w_matmul_fn;
    struct mindfour_metal_function f32_bf16w_unembedding_fn;
    struct mindfour_metal_function f32_rope_fn;
    struct mindfour_metal_function f32_mf4w_moe_matmul_swiglu_fn;
    struct mindfour_metal_function f32_mf4w_moe_matmul_fn;
    struct mindfour_metal_function f32_accumulate_e4_fn;
    struct mindfour_metal_function f32_topk_softmax_e32_k4_fn;
    struct mindfour_metal_function f32_topk_softmax_e128_k4_fn;
    struct mindfour_metal_function f32_sdpa_q8_d64_fn;
    struct mindfour_metal_function f32_softmax_fn;

    size_t per_block_shared_weights_size;
    size_t per_expert_block_weight_size;

    size_t attn_rmsnorm_gain_offset;
    size_t attn_qkv_weight_offset;
    size_t attn_qkv_bias_offset;
    size_t attn_sdpa_sink_offset;
    size_t attn_out_weight_offset;
    size_t attn_out_bias_offset;
    size_t mlp_rmsnorm_gain_offset;
    size_t mlp_gate_weight_offset;
    size_t mlp_gate_bias_offset;
    size_t mlp_swiglu_scale_offset;
    size_t mlp_swiglu_bias_offset;
    size_t mlp_out_block_offset;
    size_t mlp_out_scale_offset;
    size_t mlp_out_bias_offset;
    size_t rmsnorm_weight_offset;
    size_t unembedding_weight_offset;

    struct mindfour_metal_buffer shared_weight_buffer;

    struct mindfour_metal_buffer block_weight_buffers[];
};

#define mindfour_DEFAULT_BATCH_SIZE 128

struct mindfour_context
{
    atomic_uint_least64_t ref_count;

    struct mindfour_model *model;

    size_t num_tokens;

    size_t num_kv_tokens;

    size_t max_tokens;

    size_t kvcache_size;
    size_t allocation_size;

    struct mindfour_metal_buffer residual_activation_buffer;
    struct mindfour_metal_buffer rmsnorm_activation_buffer;
    struct mindfour_metal_buffer qkv_activation_buffer;
    struct mindfour_metal_buffer sdpa_activation_buffer;
    struct mindfour_metal_buffer gate_activation_buffer;
    struct mindfour_metal_buffer expert_activation_buffer;
    struct mindfour_metal_buffer swiglu_activation_buffer;
    struct mindfour_metal_buffer moe_activation_buffer;

    struct mindfour_metal_buffer token_buffer;
    struct mindfour_metal_buffer score_buffer;
    struct mindfour_metal_buffer prob_buffer;
    struct mindfour_metal_buffer sum_buffer;
    struct mindfour_metal_buffer argmax_buffer;
    struct mindfour_metal_buffer kvcache_buffer;
};

struct mindfour_sampler
{
    atomic_uint_least64_t ref_count;

    float temperature;
    float top_p;
    float presence_penalty;
    float frequency_penalty;
};
