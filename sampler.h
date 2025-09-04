// sampler.h

#pragma once

#include <stdlib.h>
#include <stddef.h>
#include <math.h>

#include "config.h"

typedef struct
{
    float prob;
    int index;
} ProbIndex;

typedef struct
{
    float temperature;
    float topp;
    int top_k;
    unsigned long long rng_state;
    ProbIndex* probindex;
} Sampler;

// random number generation
static inline unsigned int 
random_u32(unsigned long long *state)
{
    *state ^= *state >> 12; *state ^= *state << 25; *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

static inline float 
random_f32(unsigned long long *state)
{
    return (random_u32(state) >> 8) / 16777216.0f;
}

// comparison function for qsort
static int 
compare_prob_desc(
        const void* a, 
        const void* b)
{
    ProbIndex* pa = (ProbIndex*)a;
    ProbIndex* pb = (ProbIndex*)b;
    if (pa->prob > pb->prob) return -1;
    if (pa->prob < pb->prob) return 1;
    return 0;
}

// quickselect algorithm
static void 
quick_select(
        ProbIndex* arr, 
        int n, 
        int k)
{
    int l = 0, r = n - 1;
    while (l < r)
    {
        ProbIndex pivot = arr[k];
        int i = l, j = r;
        do
        {
            while (arr[i].prob > pivot.prob) i++;
            while (arr[j].prob < pivot.prob) j--;
            if (i <= j)
            {
                ProbIndex temp = arr[i]; arr[i] = arr[j]; arr[j] = temp;
                i++; j--;
            }
        } while (i <= j);

        if (j < k) l = i;
        if (i > k) r = j;
    }
}

// argmax for greedy sampling
static inline int 
sample_argmax(
        float* logits)
{
    int max_i = 0;
    float max_p = logits[0];
    for (int i = 1; i < VOCAB_SIZE; i++)
    {
        if (logits[i] > max_p)
        {
            max_i = i;
            max_p = logits[i];
        }
    }
    return max_i;
}

static inline void 
build_sampler(
        Sampler* sampler, 
        float temperature, 
        float topp, 
        int top_k, 
        unsigned long long rng_seed)
{
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->top_k = (top_k == 0) ? VOCAB_SIZE : top_k;
    sampler->rng_state = rng_seed;
    sampler->probindex = (ProbIndex*)malloc(VOCAB_SIZE * sizeof(ProbIndex));
}

static inline void 
free_sampler(Sampler* sampler)
{
    free(sampler->probindex);
}

static inline int 
sample(
        Sampler* sampler, 
        float* logits)
{
    if (sampler->temperature == 0.0f || sampler->top_k == 1)
    {
        return sample_argmax(logits);
    }
    for (int i = 0; i < VOCAB_SIZE; i++)
    {
        sampler->probindex[i].prob = logits[i];
        sampler->probindex[i].index = i;
    }
    
    int n_cands = (sampler->top_k < VOCAB_SIZE) ? sampler->top_k : VOCAB_SIZE;
    quick_select(sampler->probindex, VOCAB_SIZE, n_cands);
    float max_logit = sampler->probindex[0].prob;
    for (int i = 1; i < n_cands; i++)
    {
        if (sampler->probindex[i].prob > max_logit) { max_logit = sampler->probindex[i].prob; }
    }
    
    float prob_sum = 0.0f;
    for (int i = 0; i < n_cands; i++)
    {
        float prob = expf((sampler->probindex[i].prob - max_logit) / sampler->temperature);
        sampler->probindex[i].prob = prob;
        prob_sum += prob;
    }

    for (int i = 0; i < n_cands; i++) { sampler->probindex[i].prob /= prob_sum; }
    
    if (sampler->topp > 0.0f && sampler->topp < 1.0f)
    {
        qsort(sampler->probindex, n_cands, sizeof(ProbIndex), compare_prob_desc);
        float cumulative_prob = 0.0f;
        int last_idx = n_cands - 1;
        for (int i = 0; i < n_cands; i++)
        {
            cumulative_prob += sampler->probindex[i].prob;
            if (cumulative_prob > sampler->topp) { last_idx = i; break; }
        }
        n_cands = last_idx + 1;
        prob_sum = cumulative_prob;
    }

    float coin = random_f32(&sampler->rng_state) * prob_sum;
    float cdf = 0.0f;
    for (int i = 0; i < n_cands; i++)
    {
        cdf += sampler->probindex[i].prob;
        if (coin < cdf) { return sampler->probindex[i].index; }
    }
    return sampler->probindex[n_cands - 1].index;
}
