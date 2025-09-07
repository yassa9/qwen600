// main.cu

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sampler.h"
#include "tokenizer.h"
#include "qwen_model.cuh"


// ================================================================
// utils
// ================================================================

void
print_banner()
{
    printf("\n" COLOR_ORANGE R"(
   ██████╗ ██╗    ██╗███████╗███╗   ██╗ ██████╗  ██████╗  ██████╗
  ██╔═══██╗██║    ██║██╔════╝████╗  ██║██╔════╝ ██╔═████╗██╔═████╗
  ██║   ██║██║ █╗ ██║█████╗  ██╔██╗ ██║███████╗ ██║██╔██║██║██╔██║
  ██║▄▄ ██║██║███╗██║██╔══╝  ██║╚██╗██║██╔═══██╗████╔╝██║████╔╝██║
  ╚██████╔╝╚███╔███╔╝███████╗██║ ╚████║╚██████╔╝╚██████╔╝╚██████╔╝
   ╚══▀▀═╝  ╚══╝╚══╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝  ╚═════╝  ╚═════╝
)" COLOR_RESET);

    printf(COLOR_CYAN "                          github: @yassa9\n" COLOR_RESET);
}

long
time_in_ms()
{
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

void 
construct_path(
        char* out_path, 
        size_t out_size, 
        const char* dir, 
        const char* filename)
{
    size_t len = strlen(dir);
    if (len > 0 && dir[len - 1] == '/')
    {
        // directory already has a slash, so don't add another one
        snprintf(out_path, out_size, "%s%s", dir, filename);
    }
    else
    {
        // directory does not have a slash, so add one
        snprintf(out_path, out_size, "%s/%s", dir, filename);
    }
}


// ================================================================
// chat loop
// ================================================================

void
read_stdin(
    const char* guide,
    char* buffer,
    size_t bufsize)
{
    // read a line from stdin, up to but not including \n
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL)
    {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n')
        {
            // strip newline
            buffer[len - 1] = '\0';
        }
    }
}

void 
chat(
    Transformer *transformer, 
    Tokenizer *tokenizer,
    Sampler *sampler,
    char *cli_user_prompt, 
    char *system_prompt,
    int enable_thinking)
{
    char user_prompt[PROMPT_BUFFER_SIZE];
    char rendered_prompt[PROMPT_BUFFER_SIZE];
    int num_prompt_tokens = 0;
    int *prompt_tokens = (int *)malloc(PROMPT_BUFFER_SIZE * sizeof(int));

    int user_turn = 1;
    int next;
    int token;
    int pos = 0;

    long start_time = 0;
    int generated_tokens = 0;

    while (1)
    {
        if (pos >= SEQ_LEN)
        {
            printf("\n%s(context window full, clearing)%s\n", COLOR_YELLOW, COLOR_RESET);
            user_turn = 1;
            pos = 0;
        }

        if (user_turn)
        {
            if (cli_user_prompt != NULL)
            {
                if (pos > 0) break;
                strcpy(user_prompt, cli_user_prompt);
            }
            else
            {
                read_stdin("\n>> ", user_prompt, sizeof(user_prompt));
                if (!user_prompt[0]) break; // exit on empty prompt
            }

            // render the prompt with the correct template
            if (pos == 0 && system_prompt) { sprintf(rendered_prompt, tokenizer->system_prompt_template, system_prompt, user_prompt); } 
            else { sprintf(rendered_prompt, tokenizer->prompt_template, user_prompt); }

            // encode the prompt & reset the position for the new sequence
            encode(tokenizer, rendered_prompt, prompt_tokens, &num_prompt_tokens);
            pos = 0;
            user_turn = 0;
            // generated_tokens = 0;
            printf("\n");
        }

        if (pos == num_prompt_tokens) { start_time = time_in_ms(); }
        token = (pos < num_prompt_tokens) ? prompt_tokens[pos] : next;

        float *logits = forward(transformer, token, pos++);
        next = sample(sampler, logits);

        if (pos > num_prompt_tokens)
        {
            generated_tokens++;

            static int in_thinking_section = 0;
            static int in_bold_section = 0;

            if (pos == num_prompt_tokens + 1)
            {
                // first token of the response
                in_thinking_section = enable_thinking; // reset thinking state
                in_bold_section = 0; // reset bold state
                if (in_thinking_section) { printf(COLOR_YELLOW); }
            }

            char* piece = decode(tokenizer, token);

            if (strcmp(piece, "</think>") == 0)
            {
                in_thinking_section = 0;
                if (!in_bold_section) { printf(COLOR_RESET); }
            }
            else
            {
                char* current_pos = piece;
                char* marker;
                while ((marker = strstr(current_pos, "**")) != NULL)
                {
                    // print the text before the marker
                    fwrite(current_pos, 1, marker - current_pos, stdout);

                    // flip the bold state and change color accordingly
                    in_bold_section = !in_bold_section;
                    if (in_bold_section) { printf(COLOR_BOLD_RED); }
                    else if (in_thinking_section) { printf(COLOR_YELLOW); }
                    else { printf(COLOR_RESET); }
                    current_pos = marker + 2; // Move past the "**"
                }
                // print any remaining text after the last marker
                printf("%s", current_pos);
            }

            fflush(stdout);

            // stop generation if we sample an EOS token
            if (next == tokenizer->eos_token_id)
            {
                long end_time = time_in_ms();
                double elapsed_s = (double)(end_time - start_time) / 1000.0;
                double tps = (generated_tokens > 0 && elapsed_s > 0) ? (generated_tokens - 1) / elapsed_s : 0.0;
                printf("\n\n%s[%.2f tk/s, %d tokens in %.2fs]%s", COLOR_GREEN, tps, generated_tokens - 1, elapsed_s, COLOR_RESET);
                printf("\n===================================\n");
                user_turn = 1;
                continue;
            }
        }
    }
    free(prompt_tokens);
}

// ================================================================
// main
// ================================================================
void 
error_usage()
{
    fprintf(stderr, "\nusage:   ./qwen600 <model_dir> [options]\n");
    fprintf(stderr, "example: ./qwen600 <model_dir> -r 1\n");
    fprintf(stderr, "model directory must contain:\n");
    fprintf(stderr, "  - model.safetensors\n");
    fprintf(stderr, "  - tokenizer.bin\n");
    fprintf(stderr, "  - template_*.txt files\n\n");

    fprintf(stderr, "arguments:\n");
    fprintf(stderr, "----------\n");
    fprintf(stderr, "  -r <int>    reasoning mode, 0 (default) = no thinking, 1 = thinking\n");
    fprintf(stderr, "  -s <int>    random seed, default \n");
    fprintf(stderr, "  -k <int>    k value in top-k sampling, default 20\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 0.6\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1], default 0.95\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -y <string> system prompt in chat mode, default is none\n");
    fprintf(stderr, "  \n");
    exit(EXIT_FAILURE);
}

int 
main(int argc, char *argv[])
{
    print_banner();
    // default parameters
    char *model_dir = NULL;
    float temperature = 0.6f;
    float top_p = 0.95f;
    int top_k = 20;
    char *prompt = NULL;        
    unsigned long long rng_seed = 0;
    char *system_prompt = NULL; 
    int enable_thinking = 0;    

    // poor man's C argparse so we can override the defaults above from the command line
    if (argc >= 2) { model_dir = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i+=2)
    {
        if (i + 1 >= argc)          { error_usage(); } // must have arg after flag
        if (argv[i][0] != '-')      { error_usage(); } // must start with dash
        if (strlen(argv[i]) != 2)   { error_usage(); } // must be -x (one dash, one letter)

             if (argv[i][1] == 'h') { error_usage(); } // help menu
        else if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { top_p = atof(argv[i + 1]); }
        else if (argv[i][1] == 'k') { top_k = atoi(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
        else if (argv[i][1] == 'r') { enable_thinking = atoi(argv[i + 1]); }
        else { error_usage(); }
    }

    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0) temperature = 0;
    if (top_p < 0 || top_p > 1.0) top_p = 0.95;

    char model_path[1024];
    construct_path(model_path, sizeof(model_path), model_dir, "model.safetensors");

    Transformer transformer;
    build_transformer(&transformer, model_path);

    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, model_dir, enable_thinking);

    Sampler sampler;
    build_sampler(&sampler, temperature, top_p, top_k, rng_seed);

    chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, enable_thinking);

    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}
