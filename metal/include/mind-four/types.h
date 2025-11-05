#pragma once

enum mindfour_status
{
    mindfour_status_success = 0,
    mindfour_status_invalid_argument = 1,
    mindfour_status_unsupported_argument = 2,
    mindfour_status_invalid_state = 3,
    mindfour_status_io_error = 4,
    mindfour_status_insufficient_memory = 5,
    mindfour_status_insufficient_resources = 6,
    mindfour_status_unsupported_system = 7,
    mindfour_status_context_overflow = 8,
};

enum mindfour_special_token
{
    mindfour_special_token_invalid = 0,
    mindfour_special_token_return = 1,
    mindfour_special_token_start = 2,
    mindfour_special_token_message = 3,
    mindfour_special_token_end = 4,
    mindfour_special_token_refusal = 5,
    mindfour_special_token_constrain = 6,
    mindfour_special_token_channel = 7,
    mindfour_special_token_call = 8,
    mindfour_special_token_untrusted = 9,
    mindfour_special_token_end_untrusted = 10,
    mindfour_special_token_max,
};

typedef struct mindfour_model *mindfour_model_t;

typedef struct mindfour_tokenizer *mindfour_tokenizer_t;

typedef struct mindfour_context *mindfour_context_t;

typedef struct mindfour_sampler *mindfour_sampler_t;
