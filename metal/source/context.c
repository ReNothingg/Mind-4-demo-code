#include <assert.h>
#include <float.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <mind-four.h>

#include "internal/datatype.h"
#include "internal/model.h"
#include "internal/metal.h"
#include "internal/metal-kernels.h"
#include "internal/log.h"
#include "internal/rng.h"

enum mindfour_status mindfour_ABI mindfour_context_create(
    mindfour_model_t model,
    size_t context_length,
    mindfour_context_t *context_out)
{
    *context_out = NULL;

    enum mindfour_status status = mindfour_status_success;
    struct mindfour_context *context = NULL;

    if (context_length == 0)
    {
        context_length = model->context_length;
    }
    else if (context_length > model->context_length)
    {
        mindfour_LOG_ERROR("requested context length %zu exceeds model context length %" PRIu32,
                           context_length, model->context_length);
        status = mindfour_status_invalid_argument;
        goto cleanup;
    }

    context = malloc(sizeof(struct mindfour_context));
    if (context == NULL)
    {
        mindfour_LOG_ERROR("failed to allocate %zu bytes for Context object",
                           sizeof(struct mindfour_context));
        status = mindfour_status_insufficient_memory;
        goto cleanup;
    }
    memset(context, 0, sizeof(struct mindfour_context));

    atomic_store_explicit(&context->ref_count, 1, memory_order_relaxed);
    context->max_tokens = context_length;

    status = mindfour_metal_buffer_create(&model->device, model->max_batch_tokens * model->embedding_dim * sizeof(float), NULL, &context->residual_activation_buffer);
    if (status != mindfour_status_success)
    {
        goto cleanup;
    }
    status = mindfour_metal_buffer_create(&model->device, model->max_batch_tokens * model->embedding_dim * sizeof(float), NULL, &context->rmsnorm_activation_buffer);
    if (status != mindfour_status_success)
    {
        goto cleanup;
    }
    status = mindfour_metal_buffer_create(&model->device, model->max_batch_tokens * model->head_dim * (model->num_heads + 2 * model->num_kv_heads) * sizeof(float), NULL, &context->qkv_activation_buffer);
    if (status != mindfour_status_success)
    {
        goto cleanup;
    }
    status = mindfour_metal_buffer_create(&model->device, model->max_batch_tokens * model->head_dim * model->num_heads * sizeof(float), NULL, &context->sdpa_activation_buffer);
    if (status != mindfour_status_success)
    {
        goto cleanup;
    }
    status = mindfour_metal_buffer_create(&model->device, model->max_batch_tokens * model->num_experts * sizeof(float), NULL, &context->gate_activation_buffer);
    if (status != mindfour_status_success)
    {
        goto cleanup;
    }
    status = mindfour_metal_buffer_create(&model->device, model->max_batch_tokens * model->num_experts * sizeof(struct mindfour_expert_prediction), NULL, &context->expert_activation_buffer);
    if (status != mindfour_status_success)
    {
        goto cleanup;
    }
    status = mindfour_metal_buffer_create(&model->device, model->max_batch_tokens * model->num_active_experts * model->mlp_dim * sizeof(float), NULL, &context->swiglu_activation_buffer);
    if (status != mindfour_status_success)
    {
        goto cleanup;
    }
    status = mindfour_metal_buffer_create(&model->device, model->max_batch_tokens * model->num_active_experts * model->embedding_dim * sizeof(float), NULL, &context->moe_activation_buffer);
    if (status != mindfour_status_success)
    {
        goto cleanup;
    }

    status = mindfour_metal_buffer_create(&model->device, context_length * sizeof(uint32_t), NULL, &context->token_buffer);
    if (status != mindfour_status_success)
    {
        goto cleanup;
    }
    status = mindfour_metal_buffer_create(&model->device, model->max_batch_tokens * model->vocabulary_size * sizeof(float), NULL, &context->score_buffer);
    if (status != mindfour_status_success)
    {
        goto cleanup;
    }
    status = mindfour_metal_buffer_create(&model->device, model->max_batch_tokens * model->vocabulary_size * sizeof(float), NULL, &context->prob_buffer);
    if (status != mindfour_status_success)
    {
        goto cleanup;
    }
    status = mindfour_metal_buffer_create(&model->device, model->max_batch_tokens * model->max_threadgroups * sizeof(float), NULL, &context->sum_buffer);
    if (status != mindfour_status_success)
    {
        goto cleanup;
    }
    status = mindfour_metal_buffer_create(&model->device, model->max_batch_tokens * sizeof(uint64_t), NULL, &context->argmax_buffer);
    if (status != mindfour_status_success)
    {
        goto cleanup;
    }
    status = mindfour_metal_buffer_create(&model->device, model->num_blocks * context_length * 2 * model->num_kv_heads * model->head_dim * sizeof(float), NULL, &context->kvcache_buffer);
    if (status != mindfour_status_success)
    {
        goto cleanup;
    }

    context->kvcache_size = context->kvcache_buffer.size;
    context->allocation_size =
        context->residual_activation_buffer.size + context->rmsnorm_activation_buffer.size +
        context->qkv_activation_buffer.size + context->sdpa_activation_buffer.size +
        context->gate_activation_buffer.size + context->expert_activation_buffer.size + context->swiglu_activation_buffer.size + context->moe_activation_buffer.size +
        context->token_buffer.size + context->kvcache_buffer.size + context->score_buffer.size + context->argmax_buffer.size;

    context->model = model;
    mindfour_model_retain(model);
    *context_out = context;
    context = NULL;

cleanup:
    mindfour_context_release(context);
    return status;
}

enum mindfour_status mindfour_ABI mindfour_context_get_num_tokens(
    mindfour_context_t context,
    size_t *num_tokens_out)
{
    *num_tokens_out = context->num_tokens;
    return mindfour_status_success;
}

enum mindfour_status mindfour_ABI mindfour_context_get_max_tokens(
    mindfour_context_t context,
    size_t *max_tokens_out)
{
    *max_tokens_out = context->max_tokens;
    return mindfour_status_success;
}

enum mindfour_status mindfour_ABI mindfour_context_get_tokens(
    mindfour_context_t context,
    uint32_t *tokens_out,
    size_t max_tokens,
    size_t *num_tokens_out)
{
    *num_tokens_out = context->num_tokens;
    if (max_tokens < context->num_tokens)
    {
        return mindfour_status_insufficient_memory;
    }

    if (context->num_tokens != 0)
    {
        memcpy(tokens_out, context->token_buffer.ptr, context->num_tokens * sizeof(uint32_t));
    }
    return mindfour_status_success;
}

static enum mindfour_status process_tokens(
    mindfour_context_t context,
    size_t input_tokens_offset,
    size_t num_input_tokens,
    size_t num_output_tokens)
{
    assert(num_input_tokens != 0);
    assert(num_input_tokens <= context->max_batch_tokens);
    assert(num_output_tokens <= context->max_batch_tokens);
    assert(num_input_tokens >= num_output_tokens);

    enum mindfour_status status = mindfour_status_success;
    const struct mindfour_model *model = context->model;
    struct mindfour_metal_command_buffer command_buffer = {0};

    const size_t attn_qkv_dim = model->head_dim * (model->num_heads + 2 * model->num_kv_heads);

    status = mindfour_metal_command_buffer_create(&model->command_queue, &command_buffer);
    if (status != mindfour_status_success)
    {
        goto cleanup;
    }
    const size_t input_tokens_end = input_tokens_offset + num_input_tokens;
    for (size_t input_batch_start = input_tokens_offset;
         input_batch_start < input_tokens_end;
         input_batch_start += model->max_batch_tokens)
    {
        const size_t input_batch_size = math_min(model->max_batch_tokens, input_tokens_end - input_batch_start);
        const size_t input_batch_end = input_batch_start + input_batch_size;
        const size_t output_batch_size = math_sub_sat(num_output_tokens, input_tokens_end - input_batch_end);

        status = mindfour_metal_command_buffer_encode_launch_bf16_f32_embeddings(
            &command_buffer,
            &model->bf16_f32_embeddings_fn,
            512,
            &context->token_buffer,
            input_batch_start * sizeof(uint32_t),
            &model->shared_weight_buffer,
            0,
            &context->residual_activation_buffer,
            0,
            input_batch_size,
            model->embedding_dim);
        if (status != mindfour_status_success)
        {
            mindfour_LOG_ERROR("failed to encode bf16_f32_embeddings kernel launch");
            goto cleanup;
        }
        for (uint32_t n = 0; n < model->num_blocks; n++)
        {
            const bool last_block = n + 1 == model->num_blocks;
            const size_t num_block_output_tokens = last_block ? output_batch_size : input_batch_size;

            status = mindfour_metal_command_buffer_encode_launch_f32_bf16w_rmsnorm(
                &command_buffer,
                &model->f32_bf16w_rmsnorm_fn,
                &context->residual_activation_buffer,
                0,
                &model->shared_weight_buffer,
                model->attn_rmsnorm_gain_offset + model->per_block_shared_weights_size * n,
                &context->rmsnorm_activation_buffer,
                0,
                input_batch_size,
                model->embedding_dim,
                model->rmsnorm_epsilon);
            if (status != mindfour_status_success)
            {
                mindfour_LOG_ERROR("failed to encode f32_bf16w_rmsnorm kernel launch");
                goto cleanup;
            }

            status = mindfour_metal_command_buffer_encode_launch_f32_bf16w_matmul(
                &command_buffer,
                &model->f32_bf16w_matmul_fn,
                256,
                &context->rmsnorm_activation_buffer,
                0,
                &model->shared_weight_buffer,
                model->attn_qkv_weight_offset + model->per_block_shared_weights_size * n,
                &model->shared_weight_buffer,
                model->attn_qkv_bias_offset + model->per_block_shared_weights_size * n,
                &context->qkv_activation_buffer,
                0,
                input_batch_size,
                model->embedding_dim,
                attn_qkv_dim);
            if (status != mindfour_status_success)
            {
                mindfour_LOG_ERROR("failed to encode f32_bf16w_matmul kernel launch");
                goto cleanup;
            }

            status = mindfour_metal_command_buffer_encode_launch_f32_rope(
                &command_buffer,
                &model->f32_rope_fn,
                32,
                &context->qkv_activation_buffer,
                model->rope_theta,
                model->interpolation_scale,
                model->yarn_offset,
                model->yarn_scale,
                model->yarn_multiplier,
                input_batch_size,
                model->num_heads,
                model->num_kv_heads,
                model->head_dim,
                input_batch_start);
            if (status != mindfour_status_success)
            {
                mindfour_LOG_ERROR("failed to encode f32_rope kernel launch");
                goto cleanup;
            }

            for (uint32_t t = 0; t < input_batch_size; t++)
            {
                status = mindfour_metal_command_buffer_encode_copy_buffer(
                    &command_buffer,
                    &context->qkv_activation_buffer,
                    (t * attn_qkv_dim + model->num_heads * model->head_dim) * sizeof(float),
                    &context->kvcache_buffer,
                    (n * context->max_tokens + input_batch_start + t) * 2 * model->num_kv_heads * model->head_dim * sizeof(float),
                    2 * model->num_kv_heads * model->head_dim * sizeof(float));
                if (status != mindfour_status_success)
                {
                    mindfour_LOG_ERROR("failed to encode copy of token %" PRIu32 " to KV cache", t);
                    goto cleanup;
                }
            }

            if (num_block_output_tokens != 0)
            {
                status = mindfour_metal_command_buffer_encode_launch_f32_sdpa(
                    &command_buffer,
                    &model->f32_sdpa_q8_d64_fn,
                    &context->qkv_activation_buffer,
                    attn_qkv_dim * (input_batch_size - num_block_output_tokens) * sizeof(float),
                    &context->kvcache_buffer,
                    n * context->max_tokens * 2 * model->num_kv_heads * model->head_dim * sizeof(float),
                    &context->kvcache_buffer,
                    (n * context->max_tokens * 2 + 1) * model->num_kv_heads * model->head_dim * sizeof(float),
                    &model->shared_weight_buffer,
                    model->attn_sdpa_sink_offset + model->per_block_shared_weights_size * n,
                    &context->sdpa_activation_buffer,
                    0,
                    n % 2 == 0 ? model->attention_window : UINT32_MAX,
                    num_block_output_tokens,
                    input_batch_start + input_batch_size - num_block_output_tokens,
                    model->num_heads, model->num_kv_heads, model->head_dim);
                if (status != mindfour_status_success)
                {
                    mindfour_LOG_ERROR("failed to encode f32_sdpa kernel launch");
                    goto cleanup;
                }
                status = mindfour_metal_command_buffer_encode_launch_f32_bf16w_matmul_add(
                    &command_buffer,
                    &model->f32_bf16w_matmul_fn,
                    256,
                    &context->sdpa_activation_buffer,
                    0,
                    &model->shared_weight_buffer,
                    model->attn_out_weight_offset + model->per_block_shared_weights_size * n,
                    &model->shared_weight_buffer,
                    model->attn_out_bias_offset + model->per_block_shared_weights_size * n,
                    &context->residual_activation_buffer,
                    model->embedding_dim * (input_batch_size - num_block_output_tokens) * sizeof(float),
                    num_block_output_tokens,
                    model->num_heads * model->head_dim,
                    model->embedding_dim);
                if (status != mindfour_status_success)
                {
                    mindfour_LOG_ERROR("failed to encode f32_bf16w_matmul_add kernel launch");
                    goto cleanup;
                }

                status = mindfour_metal_command_buffer_encode_launch_f32_bf16w_rmsnorm(
                    &command_buffer,
                    &model->f32_bf16w_rmsnorm_fn,
                    &context->residual_activation_buffer,
                    model->embedding_dim * (input_batch_size - num_block_output_tokens) * sizeof(float),
                    &model->shared_weight_buffer,
                    model->mlp_rmsnorm_gain_offset + model->per_block_shared_weights_size * n,
                    &context->rmsnorm_activation_buffer,
                    0,
                    num_block_output_tokens,
                    model->embedding_dim,
                    model->rmsnorm_epsilon);
                if (status != mindfour_status_success)
                {
                    mindfour_LOG_ERROR("failed to encode f32_bf16w_rmsnorm kernel launch");
                    goto cleanup;
                }

                status = mindfour_metal_command_buffer_encode_launch_f32_bf16w_matmul(
                    &command_buffer,
                    &model->f32_bf16w_matmul_fn,
                    256,
                    &context->rmsnorm_activation_buffer,
                    0,
                    &model->shared_weight_buffer,
                    model->mlp_gate_weight_offset + model->per_block_shared_weights_size * n,
                    &model->shared_weight_buffer,
                    model->mlp_gate_bias_offset + model->per_block_shared_weights_size * n,
                    &context->gate_activation_buffer,
                    0,
                    num_block_output_tokens,
                    model->embedding_dim,
                    model->num_experts);
                if (status != mindfour_status_success)
                {
                    mindfour_LOG_ERROR("failed to encode f32_bf16w_matmul kernel launch");
                    goto cleanup;
                }

                const char *kernel_name = NULL;
                switch (model->num_experts)
                {
                case 32:
                    kernel_name = "f32_topk_softmax_e32_k4_fn";
                    status = mindfour_metal_command_buffer_encode_launch_f32_topk(
                        &command_buffer,
                        &model->f32_topk_softmax_e32_k4_fn,
                        &context->gate_activation_buffer, 0,
                        &context->expert_activation_buffer, 0,
                        num_block_output_tokens,
                        model->num_experts,
                        model->num_active_experts);
                    break;
                case 128:
                    kernel_name = "f32_topk_softmax_e128_k4_fn";
                    status = mindfour_metal_command_buffer_encode_launch_f32_topk(
                        &command_buffer,
                        &model->f32_topk_softmax_e128_k4_fn,
                        &context->gate_activation_buffer, 0,
                        &context->expert_activation_buffer, 0,
                        num_block_output_tokens,
                        model->num_experts,
                        model->num_active_experts);
                    break;
                default:
                    status = mindfour_status_unsupported_argument;
                    mindfour_LOG_ERROR("missing Top-K kernel for %" PRIu32 " experts", model->num_experts);
                    goto cleanup;
                }
                if (status != mindfour_status_success)
                {
                    mindfour_LOG_ERROR("failed to encode %s kernel launch", kernel_name);
                    goto cleanup;
                }

                status = mindfour_metal_command_buffer_encode_launch_f32_mf4w_moe_matmul_swiglu(
                    &command_buffer,
                    &model->f32_mf4w_moe_matmul_swiglu_fn,
                    512,
                    &context->rmsnorm_activation_buffer,
                    0,
                    &context->expert_activation_buffer,
                    0,
                    &model->block_weight_buffers[n],
                    0,
                    &model->block_weight_buffers[n],
                    model->mlp_swiglu_scale_offset,
                    &model->block_weight_buffers[n],
                    model->mlp_swiglu_bias_offset,
                    &context->swiglu_activation_buffer,
                    0,
                    model->swiglu_limit,
                    model->per_expert_block_weight_size,
                    num_block_output_tokens,
                    model->num_active_experts,
                    model->embedding_dim,
                    model->mlp_dim);
                if (status != mindfour_status_success)
                {
                    mindfour_LOG_ERROR("failed to encode f32_mf4w_moe_matmul_swiglu kernel launch");
                    goto cleanup;
                }

                status = mindfour_metal_command_buffer_encode_launch_f32_mf4w_moe_matmul(
                    &command_buffer,
                    &model->f32_mf4w_moe_matmul_fn,
                    512,
                    &context->swiglu_activation_buffer,
                    0,
                    &context->expert_activation_buffer,
                    0,
                    &model->block_weight_buffers[n],
                    model->mlp_out_block_offset,
                    &model->block_weight_buffers[n],
                    model->mlp_out_scale_offset,
                    &model->block_weight_buffers[n],
                    model->mlp_out_bias_offset,
                    &context->moe_activation_buffer,
                    0,
                    model->per_expert_block_weight_size,
                    num_block_output_tokens,
                    model->num_active_experts,
                    model->mlp_dim,
                    model->embedding_dim);
                if (status != mindfour_status_success)
                {
                    mindfour_LOG_ERROR("failed to encode f32_mf4w_moe_matmul kernel launch");
                    goto cleanup;
                }

                status = mindfour_metal_command_buffer_encode_launch_f32_accumulate(
                    &command_buffer,
                    &model->f32_accumulate_e4_fn,
                    256,
                    model->max_threadgroups,
                    &context->moe_activation_buffer,
                    0,
                    &context->expert_activation_buffer,
                    0,
                    &context->residual_activation_buffer,
                    model->embedding_dim * (input_batch_size - num_block_output_tokens) * sizeof(float),
                    model->embedding_dim,
                    num_block_output_tokens,
                    model->num_active_experts);
                if (status != mindfour_status_success)
                {
                    mindfour_LOG_ERROR("failed to encode f32_accumulate kernel launch");
                    goto cleanup;
                }
            }
        }

        if (output_batch_size != 0)
        {
            status = mindfour_metal_command_buffer_encode_launch_f32_bf16w_rmsnorm(
                &command_buffer,
                &model->f32_bf16w_rmsnorm_fn,
                &context->residual_activation_buffer,
                model->embedding_dim * (input_batch_size - output_batch_size) * sizeof(float),
                &model->shared_weight_buffer,
                model->rmsnorm_weight_offset,
                &context->rmsnorm_activation_buffer,
                0,
                output_batch_size,
                model->embedding_dim,
                model->rmsnorm_epsilon);
            if (status != mindfour_status_success)
            {
                mindfour_LOG_ERROR("failed to encode f32_bf16w_rmsnorm kernel launch");
                goto cleanup;
            }

            status = mindfour_metal_command_buffer_encode_fill_buffer(
                &command_buffer,
                &context->argmax_buffer,
                0,
                sizeof(uint64_t) * output_batch_size,
                0xFF);
            if (status != mindfour_status_success)
            {
                mindfour_LOG_ERROR("failed to encode fill buffer command");
                goto cleanup;
            }

            status = mindfour_metal_command_buffer_encode_launch_f32_bf16w_unembedding(
                &command_buffer,
                &model->f32_bf16w_unembedding_fn,
                256,
                model->max_threadgroups,
                &context->rmsnorm_activation_buffer,
                0,
                &model->shared_weight_buffer,
                model->unembedding_weight_offset,
                &context->score_buffer,
                0,
                &context->argmax_buffer,
                0,
                output_batch_size,
                model->embedding_dim,
                model->vocabulary_size);
            if (status != mindfour_status_success)
            {
                mindfour_LOG_ERROR("failed to encode f32_bf16w_unembedding kernel launch");
                goto cleanup;
            }
        }
    }

    mindfour_metal_command_buffer_commit(&command_buffer);
    mindfour_metal_command_buffer_wait_completion(&command_buffer, NULL);

cleanup:
    mindfour_metal_command_buffer_release(&command_buffer);
    return status;
}

enum mindfour_status mindfour_ABI mindfour_context_append_chars(
    mindfour_context_t context,
    const char *text,
    size_t text_length,
    size_t *num_tokens_out)
{
    enum mindfour_status status = mindfour_status_success;
    const struct mindfour_model *model = context->model;
    const struct mindfour_tokenizer *tokenizer = model->tokenizer;
    size_t num_appended_tokens = 0;
    while (text_length != 0)
    {
        if (context->num_tokens == context->max_tokens)
        {
            status = mindfour_status_context_overflow;
            break;
        }
        const char *tokens = tokenizer->tokens_ptr;
        uint32_t best_token = UINT32_MAX;
        uint32_t best_token_length = 0;
        for (size_t t = 0; t < tokenizer->num_text_tokens; t++)
        {
            uint16_t token_length;
            memcpy(&token_length, tokens, sizeof(uint16_t));
            tokens += sizeof(uint16_t);
            if (token_length <= text_length && token_length > best_token_length)
            {
                if (memcmp(text, tokens, token_length) == 0)
                {
                    if (token_length > best_token_length)
                    {
                        best_token = (uint32_t)t;
                        best_token_length = token_length;
                    }
                }
            }
            tokens += token_length;
        }

        if (best_token == UINT32_MAX)
        {
            mindfour_LOG_ERROR("failed to tokenize text \"%.*s\"", (int)text_length, text);
            return mindfour_status_invalid_argument;
        }

        uint32_t *input_tokens = (uint32_t *)context->token_buffer.ptr;
        if (context->num_kv_tokens > context->num_tokens)
        {
            if (input_tokens[context->num_tokens] != best_token)
            {
                input_tokens[context->num_tokens] = best_token;

                context->num_kv_tokens = context->num_tokens;
            }
            context->num_tokens++;
        }
        else
        {
            input_tokens[context->num_tokens++] = best_token;
        }
        num_appended_tokens++;
        text += best_token_length;
        text_length -= best_token_length;
    }
    if (num_tokens_out != NULL)
    {
        *num_tokens_out = num_appended_tokens;
    }
    return status;
}

enum mindfour_status mindfour_ABI mindfour_context_append_tokens(
    mindfour_context_t context,
    size_t num_tokens,
    const uint32_t *tokens)
{
    const struct mindfour_model *model = context->model;

    for (size_t t = 0; t < num_tokens; t++)
    {
        const uint32_t token = tokens[t];
        if (token >= model->vocabulary_size)
        {
            mindfour_LOG_ERROR("token %" PRIu32 " at index %zu is out of bounds for vocabulary size %" PRIu32,
                               token, t, context->model->vocabulary_size);
            return mindfour_status_invalid_argument;
        }
    }

    enum mindfour_status status = mindfour_status_success;
    uint32_t *input_tokens = (uint32_t *)context->token_buffer.ptr;
    while (num_tokens != 0)
    {
        if (context->num_tokens == context->max_tokens)
        {
            status = mindfour_status_context_overflow;
            break;
        }

        if (context->num_kv_tokens > context->num_tokens)
        {
            const size_t num_tokens_to_verify = math_min(context->num_kv_tokens - context->num_tokens, num_tokens);
            size_t num_verified_tokens = 0;
            for (; num_verified_tokens < num_tokens_to_verify; num_verified_tokens++)
            {
                if (input_tokens[context->num_tokens + num_verified_tokens] != tokens[num_verified_tokens])
                {

                    context->num_kv_tokens = context->num_tokens + num_verified_tokens;
                    break;
                }
            }

            context->num_tokens += num_verified_tokens;
            tokens += num_verified_tokens;
            num_tokens -= num_verified_tokens;
        }
        else
        {
            const size_t num_tokens_to_copy = math_min(context->max_tokens - context->num_tokens, num_tokens);
            memcpy(input_tokens + context->num_tokens, tokens, num_tokens_to_copy * sizeof(uint32_t));
            context->num_tokens += num_tokens_to_copy;
            tokens += num_tokens_to_copy;
            num_tokens -= num_tokens_to_copy;
        }
    }

    return status;
}

enum mindfour_status mindfour_ABI mindfour_context_process(
    mindfour_context_t context)
{
    if (context->num_tokens > context->num_kv_tokens)
    {
        enum mindfour_status status = process_tokens(
            context,
            context->num_kv_tokens,
            context->num_tokens - context->num_kv_tokens,
            0);
        if (status != mindfour_status_success)
        {
            return status;
        }

        context->num_kv_tokens = context->num_tokens;
    }

    return mindfour_status_success;
}

enum mindfour_status mindfour_ABI mindfour_context_sample(
    mindfour_context_t context,
    float temperature,
    uint64_t seed,
    uint32_t *token_out)
{
    enum mindfour_status status = mindfour_status_success;
    const struct mindfour_model *model = context->model;
    struct mindfour_metal_command_buffer command_buffer = {0};

    *token_out = UINT32_MAX;
    if (context->num_kv_tokens < context->num_tokens)
    {
        status = process_tokens(
            context,
            context->num_kv_tokens,
            context->num_tokens - context->num_kv_tokens,
            1);
        context->num_kv_tokens = context->num_tokens;
    }
    else
    {
        status = process_tokens(
            context,
            context->num_tokens - 1,
            1,
            1);
    }
    if (status != mindfour_status_success)
    {
        return status;
    }

    if (temperature == 0.0f)
    {
        const uint64_t argmax_bits = ((const uint64_t *)context->argmax_buffer.ptr)[0];
        *token_out = (uint32_t)argmax_bits;
    }
    else
    {
        assert(context->num_processed_tokens != 0);
        status = mindfour_metal_command_buffer_create(&context->model->command_queue, &command_buffer);
        if (status != mindfour_status_success)
        {
            goto cleanup;
        }

        uint32_t num_threadgroups = 0;
        uint32_t num_dims_per_threadgroup = 0;
        status = mindfour_metal_command_buffer_encode_launch_f32_softmax(
            &command_buffer,
            &model->f32_softmax_fn,
            256,
            model->max_threadgroups,
            &context->score_buffer,
            0,
            &context->argmax_buffer,
            0,
            &context->prob_buffer,
            0,
            &context->sum_buffer,
            0,
            model->vocabulary_size,
            1,
            temperature,
            &num_threadgroups,
            &num_dims_per_threadgroup);
        if (status != mindfour_status_success)
        {
            mindfour_LOG_ERROR("failed to encode f32_softmax kernel launch");
        }

        mindfour_metal_command_buffer_commit(&command_buffer);
        mindfour_metal_command_buffer_wait_completion(&command_buffer, NULL);

        const uint32_t sample_word = rng_squares32(context->num_tokens, seed + UINT64_C(0x123456789ABCDEF));
        float sample_cdf = (float)((int32_t)sample_word & INT32_C(0x00FFFFFF)) * 0x1.0p-24f;

        const float *sum_ptr = (const float *)context->sum_buffer.ptr;
        float sum = 0.0f;
        for (uint32_t i = 0; i < num_threadgroups; i++)
        {
            sum += sum_ptr[i];
        }
        sample_cdf *= sum;

        uint32_t block_idx = 0, token_idx = 0;
        if (sample_cdf == 0.0f)
        {

            sample_cdf = FLT_TRUE_MIN;
        }

        float cumsum = 0.0f;
        for (; block_idx < num_threadgroups; block_idx++)
        {
            const float new_cumsum = cumsum + sum_ptr[block_idx];
            if (new_cumsum >= sample_cdf)
            {
                break;
            }
            cumsum = new_cumsum;
        }
        if (block_idx == num_threadgroups)
        {
            block_idx -= 1;
        }

        const float *prob_ptr = (const float *)context->prob_buffer.ptr + block_idx * num_dims_per_threadgroup;
        assert(model->vocabulary_size > num_dims_per_threadgroup * block_idx);
        uint32_t num_dims_per_block = math_min(num_dims_per_threadgroup, model->vocabulary_size - num_dims_per_threadgroup * block_idx);
        for (; token_idx < num_dims_per_block; token_idx++)
        {
            const float new_cumsum = cumsum + prob_ptr[token_idx];
            if (new_cumsum >= sample_cdf)
            {
                break;
            }
            cumsum = new_cumsum;
        }
        if (token_idx == num_dims_per_block)
        {
            token_idx -= 1;
        }

        token_idx += block_idx * num_dims_per_threadgroup;

        *token_out = token_idx;

    cleanup:
        mindfour_metal_command_buffer_release(&command_buffer);
        return status;
    }

    return mindfour_status_success;
}

enum mindfour_status mindfour_ABI mindfour_context_reset(
    mindfour_context_t context)
{
    context->num_tokens = 0;

    return mindfour_status_success;
}

enum mindfour_status mindfour_ABI mindfour_context_retain(
    mindfour_context_t context)
{
    atomic_fetch_add_explicit(&context->ref_count, 1, memory_order_relaxed);
    return mindfour_status_success;
}

enum mindfour_status mindfour_ABI mindfour_context_release(
    mindfour_context_t context)
{
    if (context != NULL)
    {
        if (atomic_fetch_sub_explicit(&context->ref_count, 1, memory_order_acq_rel) == 1)
        {

            mindfour_metal_buffer_release(&context->residual_activation_buffer);
            mindfour_metal_buffer_release(&context->rmsnorm_activation_buffer);
            mindfour_metal_buffer_release(&context->qkv_activation_buffer);
            mindfour_metal_buffer_release(&context->sdpa_activation_buffer);
            mindfour_metal_buffer_release(&context->gate_activation_buffer);
            mindfour_metal_buffer_release(&context->expert_activation_buffer);
            mindfour_metal_buffer_release(&context->swiglu_activation_buffer);
            mindfour_metal_buffer_release(&context->moe_activation_buffer);

            mindfour_metal_buffer_release(&context->token_buffer);
            mindfour_metal_buffer_release(&context->score_buffer);
            mindfour_metal_buffer_release(&context->prob_buffer);
            mindfour_metal_buffer_release(&context->sum_buffer);
            mindfour_metal_buffer_release(&context->argmax_buffer);
            mindfour_metal_buffer_release(&context->kvcache_buffer);

            mindfour_model_release(context->model);

            memset(context, 0, sizeof(struct mindfour_context));
            free(context);
        }
    }
    return mindfour_status_success;
}
