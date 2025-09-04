// tokenizer.h

#pragma once

#include <string.h> 
#include <stdio.h> 

#include "config.h"

void construct_path(char* out_path, size_t out_size, const char* dir, const char* filename);

typedef struct {
    char **vocab;
    float *merge_scores;
    int vocab_size;
    unsigned int max_token_length;
    unsigned int bos_token_id;
    unsigned int eos_token_id;
    char prompt_template[1024];
    char system_prompt_template[1024];
} Tokenizer;

void load_single_template(char* buffer, size_t buffer_size, const char* dir_path, const char* filename) {
    char full_path[1024];
    construct_path(full_path, sizeof(full_path), dir_path, filename);

    memset(buffer, 0, buffer_size);
    FILE *file = fopen(full_path, "rb");
    if (!file) {
        fprintf(stderr, "Error: Couldn't load template file %s\n", full_path);
        exit(EXIT_FAILURE);
    }
    // Read up to buffer_size - 1 to ensure null termination
    fread(buffer, 1, buffer_size - 1, file);
    fclose(file);
}

void build_tokenizer(Tokenizer *t, const char *dir_path, int enable_thinking) {
    char tokenizer_path[1024];
    construct_path(tokenizer_path, sizeof(tokenizer_path), dir_path, "tokenizer.bin");

    t->vocab_size = VOCAB_SIZE;
    t->vocab = (char **)malloc(t->vocab_size * sizeof(char *));
    t->merge_scores = (float *)malloc(t->vocab_size * sizeof(float));

    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "Couldn't load tokenizer model %s\n", tokenizer_path); exit(EXIT_FAILURE); }
    fread(&t->max_token_length, sizeof(int), 1, file);
    fread(&t->bos_token_id, sizeof(int), 1, file);
    fread(&t->eos_token_id, sizeof(int), 1, file);

    int len;
    for (int i = 0; i < t->vocab_size; i++) {
        if (fread(t->merge_scores + i, sizeof(float), 1, file) != 1) {
            t->vocab[i] = (char *)malloc(1);
            t->vocab[i][0] = 0;
        } else {
            fread(&len, sizeof(int), 1, file);
            t->vocab[i] = (char *)malloc(len + 1);
            fread(t->vocab[i], 1, len, file);
            t->vocab[i][len] = 0;
        }
    }
    fclose(file);

    if (enable_thinking) {
        // load the "thinking" versions of the templates
        load_single_template(t->prompt_template, sizeof(t->prompt_template), dir_path, "template_user_thinking.txt");
        load_single_template(t->system_prompt_template, sizeof(t->system_prompt_template), dir_path, "template_system_thinking.txt");
    } else {
        // load the standard versions of the templates
        load_single_template(t->prompt_template, sizeof(t->prompt_template), dir_path, "template_user.txt");
        load_single_template(t->system_prompt_template, sizeof(t->system_prompt_template), dir_path, "template_system.txt");
    }
}

void free_tokenizer(Tokenizer *t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->merge_scores);
}

char *decode(Tokenizer *t, int token) {
    return t->vocab[token];
}

int str_lookup(char *str, char **vocab, int vocab_size) {
    // find a match for str in vocab, return its index or -1 if not found
    for (int i = 0; i < vocab_size; i++)
        if (!strcmp(str, vocab[i]))
            return i;

    return -1;
}

void encode(Tokenizer *t, char *text, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    char *str_buffer = (char*)malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    char special_token[64 + 1];

    // start at 0 tokens
    *n_tokens = 0;

    // process the raw (UTF-8) byte sequence of the input string
    for (char *c = text; *c != 0; c++) {
        int id, found_special_token = 0;

        // set the buffer to the current byte
        str_buffer[0] = *c;
        str_buffer[1] = 0;

        // special tokens begin with < and end with >. If we find a substring beginning with <
        // and ending with > and there's a token in the vocab for it, use that instead of parsing into
        // shorter tokens
        if (*c == '<') {
          int end_of_token_pos = -1;
          found_special_token = 0;
          for (int k = 0; *c != 0 && k < 64; k++) {
              if (c[k] == '>') {
                  end_of_token_pos = k;
                  break;
              }
          }

          if (end_of_token_pos != -1) {
              strncpy(special_token, c, end_of_token_pos + 1);
              special_token[end_of_token_pos + 1] = 0;

              id = str_lookup(special_token, t->vocab, t->vocab_size);
              if (id != -1) {
                  c += end_of_token_pos;
                  found_special_token = 1;
              }
          }
        }

        // not a special token, just look up the single character
        if (!found_special_token)
            id = str_lookup(str_buffer, t->vocab, t->vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            printf("Warning: unknown character code point %d in input, skipping.\n", *str_buffer);
            (*n_tokens)++;
        }
    }

    // merge the best consecutive pair each iteration
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i = 0; i < (*n_tokens - 1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
            int id = str_lookup(str_buffer, t->vocab, t->vocab_size);

            if (id != -1 && t->merge_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = t->merge_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1)
            break; // we couldn't find any more pairs to merge, so we're done

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx + 1; i < (*n_tokens - 1); i++)
            tokens[i] = tokens[i + 1];

        (*n_tokens)--; // token length decreased
    }

    free(str_buffer);
}
