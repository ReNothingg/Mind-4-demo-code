#pragma once

#include <stddef.h>
#include <stdint.h>

#include <mind-four/macros.h>
#include <mind-four/types.h>

#ifdef __cplusplus
extern "C"
{
#endif
    enum mindfour_status mindfour_ABI mindfour_model_create_from_file(
        const char *path,
        mindfour_model_t *model_out);

    enum mindfour_status mindfour_ABI mindfour_model_get_tokenizer(
        mindfour_model_t model,
        mindfour_tokenizer_t *tokenizer_out);

    enum mindfour_status mindfour_ABI mindfour_model_get_max_context_length(
        mindfour_model_t model,
        size_t *max_context_length_out);

    enum mindfour_status mindfour_ABI mindfour_model_retain(
        mindfour_model_t model);

    enum mindfour_status mindfour_ABI mindfour_model_release(
        mindfour_model_t model);

    enum mindfour_status mindfour_ABI mindfour_tokenizer_get_special_token_id(
        mindfour_tokenizer_t tokenizer,
        enum mindfour_special_token token_type,
        uint32_t *token_id_out);

    enum mindfour_status mindfour_ABI mindfour_tokenizer_get_num_text_tokens(
        mindfour_tokenizer_t tokenizer,
        uint32_t *num_text_tokens_out);

    enum mindfour_status mindfour_ABI mindfour_tokenizer_get_num_special_tokens(
        mindfour_tokenizer_t tokenizer,
        uint32_t *num_special_tokens_out);

    enum mindfour_status mindfour_ABI mindfour_tokenizer_get_num_tokens(
        mindfour_tokenizer_t tokenizer,
        uint32_t *num_tokens_out);

    enum mindfour_status mindfour_ABI mindfour_tokenizer_decode(
        mindfour_tokenizer_t tokenizer,
        uint32_t token_id,
        const void **token_ptr_out,
        size_t *token_size_out);

    enum mindfour_status mindfour_ABI mindfour_tokenizer_retain(
        mindfour_tokenizer_t tokenizer);

    enum mindfour_status mindfour_ABI mindfour_tokenizer_release(
        mindfour_tokenizer_t tokenizer);

    enum mindfour_status mindfour_ABI mindfour_context_create(
        mindfour_model_t model,
        size_t context_length,
        mindfour_context_t *context_out);

    enum mindfour_status mindfour_ABI mindfour_context_get_num_tokens(
        mindfour_context_t context,
        size_t *num_tokens_out);

    enum mindfour_status mindfour_ABI mindfour_context_get_max_tokens(
        mindfour_context_t context,
        size_t *max_tokens_out);

    enum mindfour_status mindfour_ABI mindfour_context_get_tokens(
        mindfour_context_t context,
        uint32_t *tokens_out,
        size_t max_tokens,
        size_t *num_tokens_out);

    enum mindfour_status mindfour_ABI mindfour_context_append_chars(
        mindfour_context_t context,
        const char *text,
        size_t text_length,
        size_t *num_tokens_out);

    enum mindfour_status mindfour_ABI mindfour_context_append_tokens(
        mindfour_context_t context,
        size_t num_tokens,
        const uint32_t *tokens);

    enum mindfour_status mindfour_ABI mindfour_context_reset(
        mindfour_context_t context);

    enum mindfour_status mindfour_ABI mindfour_context_process(
        mindfour_context_t context);

    enum mindfour_status mindfour_ABI mindfour_context_sample(
        mindfour_context_t context,
        float temperature,
        uint64_t seed,
        uint32_t *token_out);

    enum mindfour_status mindfour_ABI mindfour_context_retain(
        mindfour_context_t context);

    enum mindfour_status mindfour_ABI mindfour_context_release(
        mindfour_context_t context);

    enum mindfour_status mindfour_ABI mindfour_sampler_create(
        mindfour_sampler_t *sampler_out);

    enum mindfour_status mindfour_ABI mindfour_sampler_set_temperature(
        mindfour_sampler_t sampler,
        float temperature);

    enum mindfour_status mindfour_ABI mindfour_sampler_set_top_p(
        mindfour_sampler_t sampler,
        float top_p);

    enum mindfour_status mindfour_ABI mindfour_sampler_set_presence_penalty(
        mindfour_sampler_t sampler,
        float presence_penalty);

    enum mindfour_status mindfour_ABI mindfour_sampler_set_frequency_penalty(
        mindfour_sampler_t sampler,
        float frequency_penalty);

    enum mindfour_status mindfour_ABI mindfour_sampler_retain(
        mindfour_sampler_t sampler);

    enum mindfour_status mindfour_ABI mindfour_sampler_release(
        mindfour_sampler_t sampler);

#ifdef __cplusplus
}
#endif
